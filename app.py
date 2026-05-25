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
    """
    Redução de ruído via espectral subtraction.
    Usa os primeiros 0.5s como amostra do ruído de fundo.
    """
    try:
        import numpy.fft as fft
        n_fft = 2048
        hop   = 512

        # Amostra de ruído: primeiros 0.5s
        n_ruido = min(int(sr * 0.5), len(y) // 4)
        ruido   = y[:n_ruido]

        # Espectro médio do ruído
        stft_ruido = librosa.stft(ruido, n_fft=n_fft, hop_length=hop)
        perfil_ruido = np.mean(np.abs(stft_ruido), axis=1, keepdims=True)

        # STFT do áudio completo
        stft_y = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag    = np.abs(stft_y)
        fase   = np.angle(stft_y)

        # Subtração espectral com fator de intensidade
        fator  = 1.0 + intensidade * 3.0  # 0.5 → 2.5x, 1.0 → 4.0x
        mag_limpa = np.maximum(mag - perfil_ruido * fator, mag * 0.05)

        # Reconstrói áudio
        stft_limpo = mag_limpa * np.exp(1j * fase)
        y_limpo = librosa.istft(stft_limpo, hop_length=hop, length=len(y))
        print(f'[ruido] redução aplicada (fator={fator:.1f})')
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

def autotune_chunk(chunk, sr, escala_midi, strength):
    hop = 256
    frame_len = 2048
    janela_s = 0.1
    janela_samples = int(janela_s * sr)

    resultado = np.zeros_like(chunk, dtype=np.float32)
    contagem  = np.zeros_like(chunk, dtype=np.float32)

    f0 = librosa.yin(chunk,
        fmin=float(librosa.note_to_hz('C2')),
        fmax=float(librosa.note_to_hz('C7')),
        sr=sr, frame_length=frame_len, hop_length=hop)

    for inicio in range(0, len(chunk) - janela_samples, janela_samples // 2):
        fim = inicio + janela_samples
        trecho = chunk[inicio:fim]

        f_ini = inicio // hop
        f_fim = fim // hop
        f0_trecho = f0[f_ini:f_fim]
        validos = f0_trecho[(f0_trecho > 80) & (f0_trecho < 1100) & ~np.isnan(f0_trecho)]

        if len(validos) == 0:
            resultado[inicio:fim] += trecho
            contagem[inicio:fim]  += 1
            continue

        pitch = float(np.median(validos))
        midi_atual = freq_para_midi(pitch)
        midi_alvo  = nota_mais_proxima(midi_atual, escala_midi)
        n_steps    = (midi_alvo - midi_atual) * strength

        if abs(n_steps) > 0.02:
            try:
                trecho_afinado = librosa.effects.pitch_shift(
                    trecho.astype(np.float32), sr=sr,
                    n_steps=n_steps, bins_per_octave=48)
                resultado[inicio:fim] += trecho_afinado
            except:
                resultado[inicio:fim] += trecho
        else:
            resultado[inicio:fim] += trecho

        contagem[inicio:fim] += 1

    mask = contagem > 0
    resultado[mask] /= contagem[mask]
    return resultado

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
            resultado.append(autotune_chunk(chunk, sr, escala_midi, strength))
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

    arq        = request.files['audio']
    tonica     = request.form.get('tonica',    'C')
    escala     = request.form.get('escala',    'maior')
    strength   = float(request.form.get('strength',  0.8))
    reducao_ruido = float(request.form.get('reducao_ruido', 0.0))

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

        # 1. Elimina ruído (se ativado)
        if reducao_ruido > 0:
            y = eliminar_ruido(y, sr, intensidade=reducao_ruido)

        # 2. Afina
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
