import os, uuid, subprocess, shutil, traceback
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_from_directory
from pedalboard import Pedalboard, Chorus, Reverb, Compressor, Gain

app = Flask(__name__)
app.config['UPLOAD_FOLDER']    = '/tmp/venenno_up'
app.config['PROCESSED_FOLDER'] = '/tmp/venenno_out'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

FFMPEG = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'

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
        [FFMPEG,'-y','-i',origem,'-ar','44100','-ac','1','-f','wav',dest],
        capture_output=True, timeout=120
    )
    if r.returncode != 0:
        raise RuntimeError('ffmpeg: ' + r.stderr.decode(errors='replace')[-400:])
    return dest

def gerar_escala(tonica, escala):
    idx = NOTAS.index(tonica)
    ivs = ESCALAS.get(escala, ESCALAS['cromatica'])
    return [(o+1)*12 + idx + iv for o in range(-1,9) for iv in ivs]

def autotune(y, sr, tonica='C', escala='cromatica', strength=1.0, smoothing=0.0):
    try:
        f0, voiced, _ = librosa.pyin(y,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=sr, frame_length=2048, hop_length=512)
    except Exception as ex:
        print('pyin erro:', ex); return y

    escala_midi = gerar_escala(tonica, escala)
    shifts = np.zeros(len(f0))
    for i, freq in enumerate(f0):
        if voiced[i] and freq and not np.isnan(freq) and freq > 0:
            midi = 69 + 12 * np.log2(max(freq,1e-9)/440.0)
            alvo = escala_midi[int(np.argmin(np.abs(np.array(escala_midi)-midi)))]
            shifts[i] = (alvo - midi) * strength

    if smoothing > 0:
        from scipy.ndimage import uniform_filter1d
        shifts = uniform_filter1d(shifts, size=max(1,int(smoothing*20)))

    vs = shifts[voiced.astype(bool)]
    if len(vs)==0 or np.all(np.isnan(vs)): return y
    st = float(np.nanmean(vs))
    if abs(st) < 0.01: return y
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=st)
    except Exception as ex:
        print('pitch_shift erro:', ex); return y

def efeitos(y, sr, reverb=0.0, chorus=False, compressor=True):
    chain = []
    if compressor: chain.append(Compressor(threshold_db=-20, ratio=4))
    if chorus:     chain.append(Chorus())
    if reverb > 0: chain.append(Reverb(room_size=float(reverb)))
    chain.append(Gain(gain_db=2))
    return Pedalboard(chain)(y.astype(np.float32), sr)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processar', methods=['POST'])
def processar():
    if 'audio' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arq  = request.files['audio']
    tonica    = request.form.get('tonica',    'C')
    escala    = request.form.get('escala',    'cromatica')
    strength  = float(request.form.get('strength',  0.5))
    smoothing = float(request.form.get('smoothing', 0.5))
    reverb    = float(request.form.get('reverb',    0.0))
    chorus    = request.form.get('chorus',    'false').lower()=='true'
    comp      = request.form.get('compressor','false').lower()=='true'

    uid  = str(uuid.uuid4())
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    wav  = None

    try:
        arq.save(orig)
        tam = os.path.getsize(orig)
        print(f'[proc] ct={arq.content_type} size={tam}')
        if tam == 0:
            return jsonify({'erro': 'Arquivo vazio, tente gravar de novo.'}), 400

        wav  = converter_wav(orig)
        y, sr = librosa.load(wav, sr=None, mono=True)
        print(f'[proc] samples={len(y)} sr={sr}')

        if len(y) == 0:
            return jsonify({'erro': 'Áudio sem conteúdo após conversão.'}), 400

        y2 = autotune(y, sr, tonica=tonica, escala=escala,
                      strength=strength, smoothing=smoothing)
        y3 = efeitos(y2, sr, reverb=reverb, chorus=chorus, compressor=comp)

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
    r = subprocess.run([FFMPEG,'-version'], capture_output=True, text=True)
    return jsonify({'ffmpeg': FFMPEG, 'ok': r.returncode==0})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
