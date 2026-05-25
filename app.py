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
        [FFMPEG,'-y','-i',origem,'-ar','44100','-ac','1','-f','wav',dest],
        capture_output=True, timeout=300
    )
    if r.returncode != 0:
        raise RuntimeError('ffmpeg: ' + r.stderr.decode(errors='replace')[-300:])
    return dest

def pitch_shift_ffmpeg(wav_in, n_steps):
    """
    Usa o ffmpeg com filtro asetrate+atempo para pitch shift
    sem degradar a qualidade — muito mais limpo que librosa.
    """
    if abs(n_steps) < 0.05:
        return wav_in

    # Fator de frequência: 2^(n_steps/12)
    fator = 2 ** (n_steps / 12.0)
    # asetrate muda o pitch, atempo corrige o tempo
    filtro = f"asetrate=44100*{fator:.6f},aresample=44100,atempo={1.0/fator:.6f}"

    wav_out = wav_in + '_shifted.wav'
    r = subprocess.run(
        [FFMPEG, '-y', '-i', wav_in,
         '-af', filtro,
         '-ar', '44100', '-ac', '1', wav_out],
        capture_output=True, timeout=300
    )
    if r.returncode != 0:
        print(f'[pitch_ffmpeg erro] {r.stderr.decode(errors="replace")[-200:]}')
        return wav_in  # fallback: retorna original

    return wav_out

def eliminar_ruido(y, sr, intensidade=0.5):
    try:
        n_fft = 2048
        hop   = 512
        n_ruido = min(int(sr * 0.5), len(y) // 4)
        ruido   = y[:n_ruido]
        stft_ruido   = librosa.stft(ruido, n_fft=n_fft, hop_length=hop)
        perfil_ruido = np.mean(np.abs(stft_ruido), axis=1, keepdims=True)
        stft_y = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag    = np.abs(stft_y)
        fase   = np.angle(stft_y)
        fator  = 1.0 + intensidade * 3.0
        mag_limpa  = np.maximum(mag - perfil_ruido * fator, mag * 0.05)
        stft_limpo = mag_limpa * np.exp(1j * fase)
        y_limpo    = librosa.istft(stft_limpo, hop_length=hop, length=len(y))
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processar', methods=['POST'])
def processar():
    if 'audio' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arq           = request.files['audio']
    tonica        = request.form.get('tonica',        'C')
    escala        = request.form.get('escala',        'maior')
    strength      = float(request.form.get('strength',      0.8))
    reducao_ruido = float(request.form.get('reducao_ruido', 0.0))

    uid  = str(uuid.uuid4())
    orig = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    wav  = None
    shifted = None

    try:
        arq.save(orig)
        if os.path.getsize(orig) == 0:
            return jsonify({'erro': 'Arquivo vazio.'}), 400

        # Converte para WAV 44100Hz
        wav = converter_wav(orig)
        y, sr = librosa.load(wav, sr=44100, mono=True)
        print(f'[proc] duracao={len(y)/sr:.1f}s sr={sr} strength={strength}')

        if len(y) == 0:
            return jsonify({'erro': 'Audio sem conteudo.'}), 400

        # 1. Elimina ruído se pedido
        if reducao_ruido > 0:
            y = eliminar_ruido(y, sr, intensidade=reducao_ruido)
            # Salva de volta pro wav temporário
            sf.write(wav, y, sr)

        # 2. Detecta pitch médio
        hop = 512
        f0 = librosa.yin(y,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=sr, frame_length=2048, hop_length=hop)

        validos = f0[(f0 > 80) & (f0 < 1100) & ~np.isnan(f0)]

        if len(validos) == 0:
            print('[autotune] sem pitch, retorna original')
            nome = f'venenno_{uid}.wav'
            sf.write(os.path.join(app.config['PROCESSED_FOLDER'], nome), y, sr)
            return jsonify({'sucesso': True, 'url': f'/download/{nome}'})

        pitch     = float(np.median(validos))
        escala_midi = gerar_escala(tonica, escala)
        midi_atual  = freq_para_midi(pitch)
        midi_alvo   = nota_mais_proxima(midi_atual, escala_midi)
        n_steps     = (midi_alvo - midi_atual) * strength

        print(f'[autotune] pitch={pitch:.1f}Hz steps={n_steps:.2f}')

        # 3. Pitch shift via ffmpeg (qualidade superior)
        shifted = pitch_shift_ffmpeg(wav, n_steps)
        y_final, _ = librosa.load(shifted, sr=44100, mono=True)

        # 4. Ganho leve
        fator = 10 ** (2.0 / 20.0)
        y_final = np.clip(y_final * fator, -1.0, 1.0).astype(np.float32)

        nome = f'venenno_{uid}.wav'
        sf.write(os.path.join(app.config['PROCESSED_FOLDER'], nome), y_final, sr)
        return jsonify({'sucesso': True, 'url': f'/download/{nome}'})

    except Exception as e:
        print('ERRO:', traceback.format_exc())
        return jsonify({'erro': str(e)}), 500
    finally:
        for f in [orig, wav, shifted]:
            if f and f != wav and os.path.exists(f):
                try: os.remove(f)
                except: pass
        if wav and os.path.exists(wav):
            try: os.remove(wav)
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
