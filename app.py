import os, uuid, subprocess, traceback
import numpy as np
import librosa
import soundfile as sf
import imageio_ffmpeg
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
    dest = origem + '_conv.wav'
    r = subprocess.run(
        [FFMPEG,'-y','-i',origem,'-ar','22050','-ac','1','-f','wav',dest],
        capture_output=True, timeout=300
    )
    if r.returncode != 0:
        raise RuntimeError('ffmpeg: ' + r.stderr.decode(errors='replace')[-300:])
    return dest

def eliminar_ruido(y, sr, intensidade=0.5):
    try:
        n_fft = 2048
        hop   = 512
        n_ruido = min(int(sr * 0.5), len(y) // 4)
        ruido   = y[:n_ruido]
        stft_ruido = librosa.stft(ruido, n_fft=n_fft, hop_length=hop)
        perfil_ruido = np.mean(np.abs(stft_ruido), axis=1, keepdims=True)
        stft_y = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag    = np.abs(stft_y)
        fase   = np.angle(stft_y)
        fator  = 1.0 + intensidade * 3.0
        mag_limpa = np.maximum(mag - perfil_ruido * fator, mag * 0.05)
        stft_limpo = mag_limpa * np.exp(1j * fase)
        y_limpo = librosa.istft(stft_limpo, hop_length=hop, length=len(y))
        return y_limpo.astype(np.float32)
    except Exception as ex:
        print(f'[ruido erro] {ex}')
        return y

def gerar_escala(tonica, escala):
    idx = NOTAS.index(tonica)
    ivs = ESCALAS.get(escala, ESCALAS['cromatica'])
    return [(o+1)*12 + idx + iv for o in range(-1, 9) for iv in ivs]

def freq_para_midi(freq):
    return 69 + 12 * np.log2(freq / 440.0)

def nota_mais_proxima(midi_val, escala_midi):
    arr = np.array(escala_midi, dtype=float)
    return escala_midi[int(np.argmin(np.abs(arr - midi_val)))]

def autotune_simples(y, sr, escala_midi, strength):
    """
    Abordagem simples e limpa:
    1. Detecta pitch médio do trecho inteiro
    2. Aplica pitch_shift uma única vez
    Sem overlap, sem picotamento, sem abafamento.
    """
    hop = 512
    frame_len = 2048

    f0 = librosa.yin(y,
        fmin=float(librosa.note_to_hz('C2')),
        fmax=float(librosa.note_to_hz('C7')),
        sr=sr, frame_length=frame_len, hop_length=hop)

    validos = f0[(f0 > 80) & (f0 < 1100) & ~np.isnan(f0)]

    if len(validos) == 0:
        print('[autotune] sem pitch detectado, retornando original')
        return y

    pitch = float(np.median(validos))
    midi_atual = freq_para_midi(pitch)
    midi_alvo  = nota_mais_proxima(midi_atual, escala_midi)
    n_steps    = (midi_alvo - midi_atual) * strength

    print(f'[autotune] pitch={pitch:.1f}Hz midi={midi_atual:.1f} alvo={midi_alvo:.1f} steps={n_steps:.2f}')

    if abs(n_steps) < 0.05:
        print('[autotune] ja afinado, nada a fazer')
        return y

    return librosa.effects.pitch_shift(
        y.astype(np.float32), sr=sr,
        n_steps=n_steps, bins_per_octave=48)

def processar_em_chunks(y, sr, tonica, escala, strength):
    chunk_size  = 30 * sr
    escala_midi = gerar_escala(tonica, escala)
    resultado   = []
    total = (len(y) + chunk_size - 1) // chunk_size

    for i, inicio in enumerate(range(0, len(y), chunk_size)):
        chunk = y[inicio:inicio + chunk_size].astype(np.float32)
        print(f'[chunk {i+1}/{total}] {len(chunk)/sr:.1f}s')
        if len(chunk) < sr:
            resultado.append(chunk)
            continue
        try:
            resultado.append(autotune_simples(chunk, sr, escala_midi, strength))
        except Exception as ex:
            print(f'[chunk erro] {ex}')
            resultado.append(chunk)

    return np.concatenate(resultado)

def aplicar_ganho(y, db=2.0):
    fator = 10 ** (db / 20.0)
    return np.clip(y * fator, -1.0, 1.0).astype(np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processar', methods=['POST'])
def processar():
    if 'audio' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arq           = request.files['audio']
    tonica        = request.form.get('tonica',         'C')
    escala        = request.form.get('escala',         'maior')
    strength      = float(request.form.get('strength',       0.8))
    reducao_ruido = float(request.form.get('reducao_ruido',  0.0))

    uid  = str(uuid.uuid4())
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    wav  = None

    try:
        arq.save(orig)
        if os.path.getsize(orig) == 0:
            return jsonify({'erro': 'Arquivo vazio.'}), 400

        wav = converter_wav(orig)
        y, sr = librosa.load(wav, sr=22050, mono=True)
        print(f'[proc] duracao={len(y)/sr:.1f}s strength={strength} ruido={reducao_ruido}')

        if len(y) == 0:
            return jsonify({'erro': 'Audio sem conteudo.'}), 400

        if reducao_ruido > 0:
            y = eliminar_ruido(y, sr, intensidade=reducao_ruido)

        y2 = processar_em_chunks(y, sr, tonica=tonica, escala=escala, strength=strength)
        y3 = aplicar_ganho(y2)

        nome = f'venenno_{uid}.wav'
        sf.write(os.path.join(app.config['PROCESSED_FOLDER'], nome), y3, sr)
        return jsonify({'sucesso': True, 'url': f'/download/{nome}'})

    except Exception as e:
        print('ERRO:', traceback.format_exc())
        return jsonify({'erro': str(e)}), 500
    finally:
        for f in [orig, wav]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass

@app.route('/download/<nome>')
def download(nome):
    return send_from_directory(app.config['PROCESSED_FOLDER'], nome)

@app.route('/debug')
def debug():
    return jsonify({'ffmpeg': FFMPEG, 'existe': os.path.isfile(FFMPEG)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
