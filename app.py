import base64
import os, uuid, subprocess, traceback
import numpy as np
import librosa
import soundfile as sf
import imageio_ffmpeg
from scipy.signal import butter, sosfilt
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER']    = '/tmp/venenno_up'
app.config['PROCESSED_FOLDER'] = '/tmp/venenno_out'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

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
    if abs(n_steps) < 0.05:
        return y
    try:
        import pyrubberband as pyrb
        result = pyrb.pitch_shift(y, sr, n_steps)
        print(f'[pyrubberband] steps={n_steps:.3f}')
        return result.astype(np.float32)
    except Exception as ex:
        print(f'[pyrubberband erro] {ex} — librosa fallback')
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=96).astype(np.float32)

@app.route('/')
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
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    tmp_files = []

    try:
        arq.save(orig); tmp_files.append(orig)
        if os.path.getsize(orig) == 0:
            return jsonify({'erro': 'Arquivo vazio.'}), 400

        wav = converter_wav(orig); tmp_files.append(wav)
        y, sr = librosa.load(wav, sr=44100, mono=True)
        print(f'[proc] dur={len(y)/sr:.1f}s strength={strength}')
        if len(y) == 0:
            return jsonify({'erro': 'Audio sem conteudo.'}), 400

        # Pipeline: ruido -> eq -> autotune
        if reducao_ruido > 0:
            y = eliminar_ruido(y, sr, intensidade=reducao_ruido)

        y = aplicar_eq(y, sr, eq_graves, eq_medios, eq_agudos)

        f0 = librosa.yin(y,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=sr, frame_length=2048, hop_length=512)
        validos = f0[(f0 > 80) & (f0 < 1100) & ~np.isnan(f0)]

        if len(validos) > 0:
            pitch = float(np.median(validos))
            escala_midi = gerar_escala(tonica, escala)
            midi_atual  = freq_para_midi(pitch)
            midi_alvo   = nota_mais_proxima(midi_atual, escala_midi)
            n_steps     = (midi_alvo - midi_atual) * strength
            print(f'[autotune] pitch={pitch:.1f}Hz steps={n_steps:.3f}')
            y = pitch_shift_melhor(y, sr, n_steps)

        fator = 10 ** (2.0 / 20.0)
        y = np.clip(y * fator, -1.0, 1.0).astype(np.float32)

        wav_out = os.path.join(app.config['PROCESSED_FOLDER'], f'tmp_{uid}.wav')
        mp3_out = os.path.join(app.config['PROCESSED_FOLDER'], f'venenno_{uid}.mp3')
        tmp_files.append(wav_out)

        sf.write(wav_out, y, sr)
        wav_para_mp3(wav_out, mp3_out)

        with open(mp3_out, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({'sucesso': True, 'url': f'/download/venenno_{uid}.mp3', 'audio_b64': audio_b64})

    except Exception as e:
        print('ERRO:', traceback.format_exc())
        return jsonify({'erro': str(e)}), 500
    finally:
        for f in tmp_files:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass

@app.route('/download/<nome>')
def download(nome):
    return send_from_directory(
        app.config['PROCESSED_FOLDER'], nome,
        as_attachment=True,
        download_name=nome
    )

@app.route('/debug')
def debug():
    return jsonify({'ffmpeg': FFMPEG, 'existe': os.path.isfile(FFMPEG)})

@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
