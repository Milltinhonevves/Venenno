import os, uuid, subprocess, shutil, traceback
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER']    = '/tmp/venenno_up'
app.config['PROCESSED_FOLDER'] = '/tmp/venenno_out'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def get_ffmpeg():
    for p in [shutil.which('ffmpeg'), '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg',
              '/nix/var/nix/profiles/default/bin/ffmpeg', '/run/current-system/sw/bin/ffmpeg']:
        if p and os.path.isfile(p):
            return p
    # tenta achar no PATH do nix
    try:
        r = subprocess.run(['which','ffmpeg'], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except: pass
    return None

NOTAS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ESCALAS = {
    'maior':             [0,2,4,5,7,9,11],
    'menor':             [0,2,3,5,7,8,10],
    'pentatonica_maior': [0,2,4,7,9],
    'pentatonica_menor': [0,3,5,7,10],
    'cromatica':         list(range(12)),
}

def converter_wav(origem):
    ffmpeg = get_ffmpeg()
    if not ffmpeg:
        raise RuntimeError('ffmpeg nao encontrado no servidor.')
    dest = origem + '_conv.wav'
    r = subprocess.run(
        [ffmpeg,'-y','-i',origem,'-ar','44100','-ac','1','-f','wav',dest],
        capture_output=True, timeout=120
    )
    if r.returncode != 0:
        raise RuntimeError('ffmpeg erro: ' + r.stderr.decode(errors='replace')[-400:])
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
    tonica    = request.form.get('tonica',    'C')
    escala    = request.form.get('escala',    'cromatica')
    strength  = float(request.form.get('strength',  0.5))
    smoothing = float(request.form.get('smoothing', 0.5))

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
            return jsonify({'erro': 'Audio sem conteudo apos conversao.'}), 400

        y2 = autotune(y, sr, tonica=tonica, escala=escala,
                      strength=strength, smoothing=smoothing)
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
    ffmpeg = get_ffmpeg()
    info = {'ffmpeg_path': ffmpeg, 'encontrado': ffmpeg is not None}
    if ffmpeg:
        r = subprocess.run([ffmpeg,'-version'], capture_output=True, text=True)
        info['version'] = r.stdout[:100]
    return jsonify(info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
