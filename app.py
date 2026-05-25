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
        raise RuntimeError('ffmpeg erro: ' + r.stderr.decode(errors='replace')[-300:])
    return dest

def gerar_escala(tonica, escala):
    idx = NOTAS.index(tonica)
    ivs = ESCALAS.get(escala, ESCALAS['cromatica'])
    return [(o+1)*12 + idx + iv for o in range(-1,9) for iv in ivs]

def nota_mais_proxima(midi, escala_midi):
    arr = np.array(escala_midi)
    return escala_midi[int(np.argmin(np.abs(arr - midi)))]

def autotune_preciso(y, sr, tonica, escala, strength):
    """
    Autotune nota-a-nota: detecta pitch frame a frame e corrige cada trecho
    """
    hop = 256
    frame = 1024
    escala_midi = gerar_escala(tonica, escala)

    # Detecta pitch frame a frame
    f0 = librosa.yin(y,
        fmin=float(librosa.note_to_hz('C2')),
        fmax=float(librosa.note_to_hz('C7')),
        sr=sr, frame_length=frame, hop_length=hop)

    # Agrupa frames em segmentos de pitch similar
    resultado = np.copy(y).astype(np.float32)
    i = 0
    while i < len(f0):
        freq = f0[i]
        # Pula frames sem voz
        if freq < 80 or freq > 1000 or np.isnan(freq):
            i += 1
            continue

        # Encontra até onde esse pitch continua (janela de 10 frames)
        j = i + 1
        while j < len(f0) and j < i + 40:
            if f0[j] < 80 or f0[j] > 1000 or np.isnan(f0[j]):
                break
            if abs(f0[j] - freq) > freq * 0.15:  # mudou de nota
                break
            j += 1

        # Calcula pitch médio do segmento
        segmento_f0 = f0[i:j]
        validos = segmento_f0[(segmento_f0 > 80) & ~np.isnan(segmento_f0)]
        if len(validos) == 0:
            i = j
            continue

        pitch_seg = float(np.median(validos))
        midi_atual = 69 + 12 * np.log2(pitch_seg / 440.0)
        midi_alvo  = nota_mais_proxima(midi_atual, escala_midi)
        n_steps    = (midi_alvo - midi_atual) * strength

        if abs(n_steps) > 0.1:
            inicio_s = i * hop
            fim_s    = min(j * hop + frame, len(y))
            trecho   = y[inicio_s:fim_s]

            if len(trecho) > frame:
                try:
                    trecho_afinado = librosa.effects.pitch_shift(
                        trecho, sr=sr, n_steps=n_steps, bins_per_octave=48)
                    # Crossfade suave nas bordas
                    fade = min(512, len(trecho) // 4)
                    if fade > 0:
                        ramp = np.linspace(0, 1, fade)
                        trecho_afinado[:fade]  *= ramp
                        trecho_afinado[-fade:] *= ramp[::-1]
                        resultado[inicio_s:inicio_s+fade] *= (1 - ramp)
                        resultado[fim_s-fade:fim_s]       *= (1 - ramp[::-1])
                    resultado[inicio_s:fim_s] += trecho_afinado
                except Exception as ex:
                    print(f'[pitch_shift erro] {ex}')
        i = j

    return resultado

def processar_em_chunks(y, sr, tonica, escala, strength):
    chunk_size = 30 * sr
    resultado = []
    total = (len(y) + chunk_size - 1) // chunk_size

    for i, inicio in enumerate(range(0, len(y), chunk_size)):
        chunk = y[inicio:inicio + chunk_size]
        print(f'[chunk {i+1}/{total}] {len(chunk)/sr:.1f}s')
        if len(chunk) < sr:
            resultado.append(chunk)
            continue
        try:
            chunk_afinado = autotune_preciso(chunk, sr, tonica, escala, strength)
            resultado.append(chunk_afinado)
        except Exception as ex:
            print(f'[chunk erro] {ex}')
            resultado.append(chunk)

    return np.concatenate(resultado)

def aplicar_ganho(y, db=2.0):
    fator = 10 ** (db / 20.0)
    return np.clip(y * fator, -1.0, 1.0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processar', methods=['POST'])
def processar():
    if 'audio' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arq      = request.files['audio']
    tonica   = request.form.get('tonica',   'C')
    escala   = request.form.get('escala',   'cromatica')
    strength = float(request.form.get('strength', 0.8))

    uid  = str(uuid.uuid4())
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    wav  = None

    try:
        arq.save(orig)
        tam = os.path.getsize(orig)
        if tam == 0:
            return jsonify({'erro': 'Arquivo vazio.'}), 400

        wav = converter_wav(orig)
        y, sr = librosa.load(wav, sr=22050, mono=True)
        duracao = len(y) / sr
        print(f'[proc] duracao={duracao:.1f}s')

        if len(y) == 0:
            return jsonify({'erro': 'Audio sem conteudo.'}), 400

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
