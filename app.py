import base64
import os, uuid, subprocess, traceback
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER']    = '/tmp/venenno_up'
app.config['PROCESSED_FOLDER'] = '/tmp/venenno_out'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Usa imageio_ffmpeg (binario bundled) com fallback pro sistema
import shutil as _shutil
import imageio_ffmpeg as _ioff
_sys_ff = _shutil.which('ffmpeg')
try:
    _img_ff = _ioff.get_ffmpeg_exe()
except Exception:
    _img_ff = None
FFMPEG = _sys_ff or _img_ff or 'ffmpeg'
print(f'[startup] ffmpeg={FFMPEG} sys={_sys_ff} imageio={_img_ff}')

NOTAS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ESCALAS = {
    'maior':             [0,2,4,5,7,9,11],
    'menor':             [0,2,3,5,7,8,10],
    'pentatonica_maior': [0,2,4,7,9],
    'pentatonica_menor': [0,3,5,7,10],
    'cromatica':         list(range(12)),
}

def converter_wav(origem):
    """Converte qualquer formato para WAV - ffmpeg primeiro, librosa como fallback"""
    dest = origem + '_conv.wav'
    # MP3 do WhatsApp/Drive: usa ffmpeg direto (mais confiavel pra MP3)
    try:
        r = subprocess.run(
            [FFMPEG,'-y','-i',origem,'-ar','44100','-ac','1','-acodec','pcm_s16le','-f','wav',dest],
            capture_output=True, timeout=300
        )
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f'[convert ffmpeg] OK {os.path.getsize(dest)}b')
            return dest
        print(f'[convert ffmpeg falhou] {r.stderr.decode(errors="replace")[-200:]}')
    except Exception as e1:
        print(f'[convert ffmpeg excecao] {e1}')
    # Fallback: librosa
    try:
        y, sr = librosa.load(origem, sr=44100, mono=True)
        sf.write(dest, y.astype(np.float32), sr)
        print(f'[convert librosa] OK')
        return dest
    except Exception as e2:
        raise RuntimeError(f'Nao foi possivel converter: {e2}')


def wav_para_mp3(wav_path, mp3_path):
    r = subprocess.run(
        [FFMPEG,'-y','-i',wav_path,'-codec:a','libmp3lame','-qscale:a','2',mp3_path],
        capture_output=True, timeout=120
    )
    if r.returncode != 0:
        raise RuntimeError('ffmpeg mp3: ' + r.stderr.decode(errors='replace')[-200:])

def aplicar_eq(y, sr, graves=0.0, medios=0.0, agudos=0.0):
    """Equalizador 3 bandas direto no array numpy - sem I/O de arquivo"""
    try:
        graves = float(graves); medios = float(medios); agudos = float(agudos)
        if graves == 0 and medios == 0 and agudos == 0:
            return y

        nyq = sr / 2.0

        # Graves: shelf abaixo de 300Hz
        if graves != 0:
            gain = 10 ** (graves / 20.0)
            sos = butter(2, 300 / nyq, btype='low', output='sos')
            baixos = sosfilt(sos, y)
            altos  = y - baixos
            y = baixos * gain + altos

        # Medios: banda 500-3000Hz
        if medios != 0:
            gain = 10 ** (medios / 20.0)
            sos_l = butter(2, 500  / nyq, btype='high', output='sos')
            sos_h = butter(2, 3000 / nyq, btype='low',  output='sos')
            band = sosfilt(sos_h, sosfilt(sos_l, y))
            resto = y - band
            y = band * gain + resto

        # Agudos: shelf acima de 4000Hz
        if agudos != 0:
            gain = 10 ** (agudos / 20.0)
            sos = butter(2, 4000 / nyq, btype='high', output='sos')
            agud  = sosfilt(sos, y)
            resto = y - agud
            y = agud * gain + resto

        # Normaliza pra evitar clipping
        pico = np.max(np.abs(y))
        if pico > 1.0:
            y = y / pico

        return y.astype(np.float32)
    except Exception as e:
        print(f'[eq excecao] {e}')
        return y

def eliminar_ruido(y, sr, intensidade=0.5):
    try:
        n_fft = 2048; hop = 512
        n_ruido = min(int(sr * 0.5), len(y) // 4)
        stft_ruido   = librosa.stft(y[:n_ruido], n_fft=n_fft, hop_length=hop)
        perfil_ruido = np.mean(np.abs(stft_ruido), axis=1, keepdims=True)
        stft_y = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag  = np.abs(stft_y); fase = np.angle(stft_y)
        fator = 1.0 + intensidade * 3.0
        mag_limpa = np.maximum(mag - perfil_ruido * fator, mag * 0.05)
        return librosa.istft(mag_limpa * np.exp(1j * fase), hop_length=hop, length=len(y)).astype(np.float32)
    except Exception as ex:
        print(f'[ruido erro] {ex}'); return y

def gerar_escala(tonica, escala):
    idx = NOTAS.index(tonica)
    ivs = ESCALAS.get(escala, ESCALAS['cromatica'])
    return [(o+1)*12 + idx + iv for o in range(-1, 9) for iv in ivs]

def freq_para_midi(freq):
    return 69 + 12 * np.log2(freq / 440.0)

def nota_mais_proxima(midi_val, escala_midi):
    arr = np.array(escala_midi, dtype=float)
    return escala_midi[int(np.argmin(np.abs(arr - midi_val)))]

def pitch_shift_melhor(y, sr, n_steps):
    """Pitch shift usando librosa (confiavel no Railway)."""
    if abs(n_steps) < 0.05:
        return y
    print(f'[pitch_shift] steps={n_steps:.3f}')
    result = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=float(n_steps))
    return result.astype(np.float32)


def autotune_frame_a_frame(y, sr, tonica, escala, strength=0.8):
    """
    Autotune frame a frame leve: detecta pitch com pyin globalmente,
    divide em chunks de 500ms e corrige cada um individualmente.
    Equilibrio entre precisao e uso de memoria no Railway.
    """
    escala_midi = gerar_escala(tonica, escala)
    chunk_size  = int(sr * 0.5)   # chunks de 500ms
    hop_length  = 512
    frame_length = 2048

    # 1. Detecta pitch global com pyin
    try:
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
        validos = f0[voiced & ~np.isnan(f0) & (f0 > 60) & (f0 < 1200)]
        print(f'[autotune ff] frames_vozeados={np.sum(voiced)} validos={len(validos)}')
    except Exception as e:
        print(f'[autotune ff] pyin falhou: {e}')
        return y

    if len(validos) == 0:
        print('[autotune ff] nenhuma voz detectada')
        return y

    # 2. Para cada chunk de 500ms, calcula pitch local e corrige
    out_chunks = []
    n_chunks = max(1, int(np.ceil(len(y) / chunk_size)))

    for i in range(n_chunks):
        start = i * chunk_size
        end   = min(start + chunk_size, len(y))
        seg   = y[start:end].copy()

        # Frames pyin correspondentes a este chunk
        f_start = int(start / hop_length)
        f_end   = int(end   / hop_length)
        f0_seg  = f0[f_start:f_end] if f_end <= len(f0) else f0[f_start:]
        v_seg   = voiced[f_start:f_end] if f_end <= len(voiced) else voiced[f_start:]

        f0_local = f0_seg[v_seg & ~np.isnan(f0_seg) & (f0_seg > 60) & (f0_seg < 1200)]

        if len(f0_local) > 0:
            pitch_local = float(np.median(f0_local))
            midi_atual  = freq_para_midi(pitch_local)
            midi_alvo   = nota_mais_proxima(midi_atual, escala_midi)
            n_steps     = (midi_alvo - midi_atual) * strength
            print(f'[autotune ff] chunk={i} pitch={pitch_local:.1f}Hz steps={n_steps:.3f}')
            if abs(n_steps) > 0.05:
                try:
                    seg = librosa.effects.pitch_shift(
                        y=seg, sr=sr, n_steps=float(n_steps)
                    ).astype(np.float32)
                except Exception as ex:
                    print(f'[autotune ff] pitch_shift erro chunk {i}: {ex}')

        out_chunks.append(seg)

    return np.concatenate(out_chunks).astype(np.float32)
def index():
    return render_template('index.html')

@app.route('/processar', methods=['POST'])
def processar():
    if 'audio' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arq           = request.files['audio']
    tonica        = request.form.get('tonica', 'C')
    escala        = request.form.get('escala', 'maior')
    strength      = float(request.form.get('strength', 0.8))
    reducao_ruido = float(request.form.get('reducao_ruido', 0.0))
    eq_graves     = float(request.form.get('eq_graves', 0.0))
    eq_medios     = float(request.form.get('eq_medios', 0.0))
    eq_agudos     = float(request.form.get('eq_agudos', 0.0))

    uid = str(uuid.uuid4())
    # Detecta extensao do arquivo pelo nome ou content-type
    nome_orig = arq.filename or ''
    ext_orig  = os.path.splitext(nome_orig)[1].lower()
    if not ext_orig:
        ct = (arq.content_type or '').lower()
        if 'mp3' in ct or 'mpeg' in ct:   ext_orig = '.mp3'
        elif 'mp4' in ct or 'm4a' in ct:  ext_orig = '.mp4'
        elif 'ogg' in ct:                 ext_orig = '.ogg'
        elif 'webm' in ct:                ext_orig = '.webm'
        elif 'wav' in ct:                 ext_orig = '.wav'
        elif 'flac' in ct:                ext_orig = '.flac'
        elif 'aac' in ct:                 ext_orig = '.aac'
        else:                             ext_orig = '.mp3'  # assume mp3 por padrao
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid + ext_orig)
    tmp_files = []
    print(f'[upload] nome={nome_orig} ct={arq.content_type} ext={ext_orig}')

    try:
        arq.save(orig); tmp_files.append(orig)
        if os.path.getsize(orig) == 0:
            return jsonify({'erro': 'Arquivo vazio.'}), 400

        wav = converter_wav(orig); tmp_files.append(wav)
        # Usa 22050 Hz pra processar mais rapido (metade dos dados)
        y, sr = librosa.load(wav, sr=22050, mono=True)
        print(f'[proc] dur={len(y)/sr:.1f}s sr={sr} strength={strength}')
        if len(y) == 0:
            return jsonify({'erro': 'Audio sem conteudo.'}), 400

        # Pipeline: ruido -> eq -> autotune
        if reducao_ruido > 0:
            y = eliminar_ruido(y, sr, intensidade=reducao_ruido)

        y = aplicar_eq(y, sr, eq_graves, eq_medios, eq_agudos)

        # Autotune frame a frame (estilo Melodyne)
        y = autotune_frame_a_frame(y, sr, tonica, escala, strength)
