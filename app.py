import os, uuid, subprocess, traceback
import numpy as np
import librosa
import soundfile as sf
import imageio_ffmpeg
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER']    = '/tmp/venenno_up'
app.config['PROCESSED_FOLDER'] = '/tmp/venenno_out'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

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
        capture_output=True, timeout=120
    )
    if r.returncode != 0:
        raise RuntimeError('ffmpeg erro: ' + r.stderr.decode(errors='replace')[-300:])
    return dest

def gerar_escala(tonica, escala):
    idx = NOTAS.index(tonica)
    ivs = ESCALAS.get(escala, ESCALAS['cromatica'])
    return [(o+1)*12 + idx + iv for o in range(-1,9) for iv in ivs]

def processar_em_chunks(y, sr, tonica, escala, strength):
    """Processa audio em pedacos de 20s para nao travar memoria"""
    chunk_size = 20 * sr
    escala_midi = gerar_escala(tonica, escala)
    resultado = []

    for inicio in range(0, len(y), chunk_size):
        chunk = y[inicio:inicio + chunk_size]
        if len(chunk) < sr:  # chunk muito pequeno, adiciona sem processar
            resultado.append(chunk)
            continue

        try:
            f0 = librosa.yin(chunk,
                fmin=float(librosa.note_to_hz('C2')),
                fmax=float(librosa.note_to_hz('C7')),
                sr=sr, frame_length=1024, hop_length=256)

            validos = f0[(f0 > 80) & (f0 < 1000) & ~np.isnan(f0)]
            if len(validos) == 0:
                resultado.append(chunk)
                continue

            pitch_medio = float(np.median(validos))
            midi_atual = 69 + 12 * np.log2(pitch_medio / 440.0)
            midi_alvo = escala_midi[int(np.argmin(np.abs(np.array(escala_midi) - midi_atual)))]
            n_steps = (midi_alvo - midi_atual) * strength

            if abs(n_steps) < 0.05:
                resultado.append(chunk)
                continue

            print(f'[chunk {inicio//sr}s] pitch={pitch_medio:.1f}Hz steps={n_steps:.2f}')
            chunk_afinado = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=n_steps, bins_per_octave=24)
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

    arq  = request.files['audio']
    tonica   = request.form.get('tonica',   'C')
    escala   = request.form.get('escala',   'cromatica')
    strength = float(request.form.get('strength', 0.5))

    uid  = str(uuid.uuid4())
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    wav  = None

    try:
        arq.save(orig)
        tam = os.path.getsize(orig)
        print(f'[proc] ct={arq.content_type} size={tam}')
        if tam == 0:
            return jsonify({'erro': 'Arquivo vazio, tente gravar de novo.'}), 400

        wav = converter_wav(orig)
        y, sr = librosa.load(wav, sr=22050, mono=True)
        print(f'[proc] duracao={len(y)/sr:.1f}s samples={len(y)} sr={sr}')

        if len(y) == 0:
            return jsonify({'erro': 'Audio sem conteudo.'}), 400

        y2 = processar_em_chunks(y, sr, tonica=tonica, escala=escala, strength=strength)
        y3 = aplicar_ganho(y2)

        nome = f'venenno_{uid}.wav'
        sf.write(os.path.join(app.config['PROCESSED_FOLDER'], nome), y3, sr)
        print(f'[proc] salvo {nome} ({len(y3)/sr:.1f}s)')
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
