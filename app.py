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
    Autotune estilo Melodyne: detecta e corrige o pitch frame a frame.
    Divide o audio em janelas de 50ms, detecta pitch de cada uma,
    corrige para a nota mais proxima da escala escolhida.
    """
    hop = int(sr * 0.05)   # janela de 50ms
    win = hop * 2          # overlap de 50%
    escala_midi = gerar_escala(tonica, escala)

    # pyin frame a frame
    try:
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=sr,
            frame_length=win,
            hop_length=hop
        )
    except Exception as e:
        print(f'[autotune ff] pyin falhou: {e}')
        return y

    n_frames = len(f0)
    print(f'[autotune ff] {n_frames} frames, voiced={np.sum(voiced)}')

    if np.sum(voiced) == 0:
        print('[autotune ff] nenhuma voz detectada')
        return y

    # Cria saida com overlap-add
    out = np.zeros_like(y)
    counts = np.zeros_like(y)
    window = np.hanning(win)

    for i in range(n_frames):
        start = i * hop
        end   = start + win
        if end > len(y):
            break

        seg = y[start:end].copy()

        if voiced[i] and not np.isnan(f0[i]) and f0[i] > 60:
            midi_atual = freq_para_midi(f0[i])
            midi_alvo  = nota_mais_proxima(midi_atual, escala_midi)
            n_steps    = (midi_alvo - midi_atual) * strength

            if abs(n_steps) > 0.05:
                try:
                    seg = librosa.effects.pitch_shift(
                        y=seg, sr=sr, n_steps=float(n_steps)
                    ).astype(np.float32)
                except Exception:
                    pass  # mantém seg original se falhar

        seg_win = seg * window
        out[start:end]    += seg_win
        counts[start:end] += window

    # Normaliza onde houve sobreposição
    mask = counts > 1e-6
    out[mask] /= counts[mask]

    return out.astype(np.float32)
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

        # Detecta pitch da voz com pyin (mais preciso)
        try:
            f0, voiced_flag, _ = librosa.pyin(y,
                fmin=float(librosa.note_to_hz('C2')),
                fmax=float(librosa.note_to_hz('C7')),
                sr=sr, frame_length=2048, hop_length=512)
            validos = f0[voiced_flag & ~np.isnan(f0) & (f0 > 60) & (f0 < 1200)]
            print(f'[pyin] frames_validos={len(validos)}')
        except Exception as ep:
            print(f'[pyin erro] {ep} — fallback yin')
            # Autotune frame a frame (estilo Melodyne)
        print('[autotune ff] iniciando...')
        y = autotune_frame_a_frame(y, sr_a, tonica, escala, strength)
        print('[autotune ff] concluido')
            fator = 10 ** (2.0 / 20.0)
            y = np.clip(y * fator, -1.0, 1.0).astype(np.float32)
            wav_out = os.path.join(app.config['PROCESSED_FOLDER'], f'tmp_{uid}.wav')
            mp3_out = os.path.join(app.config['PROCESSED_FOLDER'], f'venenno_{uid}.mp3')
            tmp.append(wav_out)
            sf.write(wav_out, y, sr_audio)
            wav_para_mp3(wav_out, mp3_out)
            with open(mp3_out, 'rb') as ff:
                audio_b64 = base64.b64encode(ff.read()).decode('utf-8')
            try: os.remove(mp3_out)
            except: pass
            _jobs[uid] = {'status': 'done', 'audio_b64': audio_b64}
            print(f'[bg] job={uid} concluido b64={len(audio_b64)}')
        except Exception as e:
            import traceback
            print(f'[bg] job={uid} ERRO:\n{traceback.format_exc()}')
            _jobs[uid] = {'status': 'error', 'erro': str(e)}
        finally:
            for f in tmp + [orig]:
                try: os.remove(f)
                except: pass
    _threading.Thread(target=_bg, daemon=True).start()
    return jsonify({'job_id': uid})

@app.route('/status/<job_id>')
def status(job_id):
    job = _jobs.get(job_id, {'status': 'not_found'})
    if job['status'] in ('done', 'error', 'not_found'):
        _jobs.pop(job_id, None)
    return jsonify(job)


@app.route('/enviar', methods=['POST'])
def enviar():
    """Recebe audio como base64 em JSON, processa em background"""
    data = request.get_json(force=True, silent=True) or {}
    audio_b64_in = data.get('audio')
    if not audio_b64_in:
        return jsonify({'erro': 'Nenhum audio enviado.'}), 400

    tonica        = data.get('tonica', 'C')
    escala        = data.get('escala', 'maior')
    strength      = float(data.get('strength', 0.8))
    reducao_ruido = float(data.get('reducao_ruido', 0.0))
    semitons_extra = float(data.get('semitons_extra', 0))
    eq_graves     = float(data.get('eq_graves', 0.0))
    eq_medios     = float(data.get('eq_medios', 0.0))
    eq_agudos     = float(data.get('eq_agudos', 0.0))
    mime          = data.get('mime', 'audio/mpeg')

    # Detecta extensao pelo mime
    if 'mp3' in mime or 'mpeg' in mime:   ext = '.mp3'
    elif 'mp4' in mime or 'm4a' in mime:  ext = '.mp4'
    elif 'ogg' in mime:                   ext = '.ogg'
    elif 'webm' in mime:                  ext = '.webm'
    elif 'wav' in mime:                   ext = '.wav'
    else:                                 ext = '.mp3'

    uid  = str(uuid.uuid4())
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid + ext)

    try:
        audio_bytes = base64.b64decode(audio_b64_in)
        with open(orig, 'wb') as f:
            f.write(audio_bytes)
        print(f'[enviar] job={uid} mime={mime} ext={ext} size={len(audio_bytes)}')
    except Exception as e:
        return jsonify({'erro': f'Falha ao decodificar audio: {e}'}), 400

    _jobs[uid] = {'status': 'processing'}

    def _bg2():
        tmp = []
        try:
            wav = converter_wav(orig); tmp.append(wav)
            y, sr_a = librosa.load(wav, sr=22050, mono=True)
            if len(y) == 0:
                _jobs[uid] = {'status': 'error', 'erro': 'Audio sem conteudo.'}
                return
            if reducao_ruido > 0:
                y = eliminar_ruido(y, sr_a, intensidade=reducao_ruido)
            y = aplicar_eq(y, sr_a, eq_graves, eq_medios, eq_agudos)
            f0 = librosa.yin(y,
                fmin=float(librosa.note_to_hz('C2')),
                fmax=float(librosa.note_to_hz('C7')),
                sr=sr_a, frame_length=1024, hop_length=256)
            validos = f0[(f0 > 80) & (f0 < 1100) & ~np.isnan(f0)]
            if len(validos) > 0:
                pitch       = float(np.median(validos))
                escala_midi = gerar_escala(tonica, escala)
                midi_atual  = freq_para_midi(pitch)
                midi_alvo   = nota_mais_proxima(midi_atual, escala_midi)
                n_steps     = (midi_alvo - midi_atual) * strength
                print(f'[enviar bg] pitch={pitch:.1f}Hz steps={n_steps:.3f}')
                y = pitch_shift_melhor(y, sr_a, n_steps)
            if semitons_extra != 0:
                y = pitch_shift_melhor(y, sr_a, semitons_extra)
            fator = 10 ** (2.0 / 20.0)
            y = np.clip(y * fator, -1.0, 1.0).astype(np.float32)
            wav_out = os.path.join(app.config['PROCESSED_FOLDER'], f'tmp_{uid}.wav')
            mp3_out = os.path.join(app.config['PROCESSED_FOLDER'], f'venenno_{uid}.mp3')
            tmp.append(wav_out)
            sf.write(wav_out, y, sr_a)
            wav_para_mp3(wav_out, mp3_out)
            with open(mp3_out, 'rb') as ff:
                audio_b64_out = base64.b64encode(ff.read()).decode('utf-8')
            try: os.remove(mp3_out)
            except: pass
            _jobs[uid] = {'status': 'done', 'audio_b64': audio_b64_out}
            print(f'[enviar] job={uid} concluido b64={len(audio_b64_out)}')
        except Exception as e:
            import traceback
            print(f'[enviar] ERRO:\n{traceback.format_exc()}')
            _jobs[uid] = {'status': 'error', 'erro': str(e)}
        finally:
            for f in tmp + [orig]:
                try: os.remove(f)
                except: pass

    _threading.Thread(target=_bg2, daemon=True).start()
    return jsonify({'job_id': uid})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
