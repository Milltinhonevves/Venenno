import base64
import os, uuid, subprocess, traceback
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample as scipy_resample
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER']    = '/tmp/venenno_up'
app.config['PROCESSED_FOLDER'] = '/tmp/venenno_out'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

import shutil as _shutil
import imageio_ffmpeg as _ioff
_sys_ff = _shutil.which('ffmpeg')
try:
    _img_ff = _ioff.get_ffmpeg_exe()
except Exception:
    _img_ff = None
FFMPEG = _sys_ff or _img_ff or 'ffmpeg'
print(f'[startup] ffmpeg={FFMPEG}')

SR = 16000   # sample rate interno — minimo para voz, economiza memoria

NOTAS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ESCALAS = {
    'maior':             [0,2,4,5,7,9,11],
    'menor':             [0,2,3,5,7,8,10],
    'pentatonica_maior': [0,2,4,7,9],
    'pentatonica_menor': [0,3,5,7,10],
    'cromatica':         list(range(12)),
}

# ── Conversão ──────────────────────────────────────────────────────────────
def converter_wav(origem):
    dest = origem + '_conv.wav'
    try:
        r = subprocess.run(
            [FFMPEG,'-y','-i',origem,f'-ar',str(SR),'-ac','1','-acodec','pcm_s16le','-f','wav',dest],
            capture_output=True, timeout=120)
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest
        print(f'[ffmpeg err] {r.stderr.decode(errors="replace")[-200:]}')
    except Exception as e:
        print(f'[ffmpeg exc] {e}')
    return origem

# ── Áudio helpers ───────────────────────────────────────────────────────────
def carregar_audio(path):
    """Carrega WAV como float32 mono em SR."""
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        n = int(len(y) * SR / sr)
        y = scipy_resample(y, n).astype(np.float32)
    return y

def wav_para_mp3(wav_path, mp3_path):
    subprocess.run(
        [FFMPEG,'-y','-i',wav_path,'-codec:a','libmp3lame','-qscale:a','4',mp3_path],
        capture_output=True, timeout=120)

# ── Equalizer ──────────────────────────────────────────────────────────────
def butter_filter(y, cutoff, btype, order=4):
    nyq = SR / 2.0
    Wn  = np.clip(cutoff / nyq, 0.001, 0.999)
    if isinstance(Wn, (list,tuple)):
        Wn = [np.clip(w, 0.001, 0.999) for w in Wn]
    sos = butter(order, Wn, btype=btype, output='sos')
    return sosfilt(sos, y).astype(np.float32)

def aplicar_eq(y, graves_db, medios_db, agudos_db):
    resultado = y.copy()
    if abs(graves_db) > 0.1:
        low   = butter_filter(y, 300, 'low')
        fator = 10 ** (float(graves_db) / 20.0)
        resultado += low * (fator - 1.0)
    if abs(medios_db) > 0.1:
        mid   = butter_filter(y, [500, 3000], 'bandpass')
        fator = 10 ** (float(medios_db) / 20.0)
        resultado += mid * (fator - 1.0)
    if abs(agudos_db) > 0.1:
        high  = butter_filter(y, 4000, 'high')
        fator = 10 ** (float(agudos_db) / 20.0)
        resultado += high * (fator - 1.0)
    return np.clip(resultado, -1.0, 1.0).astype(np.float32)

# ── Pitch ───────────────────────────────────────────────────────────────────
def freq_para_midi(f):
    return 69 + 12 * np.log2(max(f, 1e-6) / 440.0)

def gerar_escala(tonica, nome_escala):
    base   = NOTAS.index(tonica) if tonica in NOTAS else 0
    graus  = ESCALAS.get(nome_escala, ESCALAS['maior'])
    return [(base + g) % 12 for g in graus]

def nota_mais_proxima(midi, escala_midi):
    oitava = int(midi) // 12
    nota   = round(midi) % 12
    candidatos = []
    for g in escala_midi:
        for ov in [oitava - 1, oitava, oitava + 1]:
            candidatos.append(ov * 12 + g)
    return min(candidatos, key=lambda m: abs(m - midi))

def yin_simples(seg):
    """
    YIN compacto: opera em janela de 2048 amostras do centro do chunk.
    Rapido, pouca memoria, suficiente para detectar pitch vocal.
    """
    # Usa janela de 2048 amostras do meio do chunk
    W = 2048
    n = len(seg)
    if n < W:
        trecho = seg
    else:
        mid = n // 2
        trecho = seg[max(0, mid - W//2): mid + W//2]

    n = len(trecho)
    if n < 256:
        return 0.0

    min_tau = max(4, int(SR / 1200))
    max_tau = min(n // 2, int(SR / 60))
    if min_tau >= max_tau:
        return 0.0

    # Difference function simples (vetorizada por slices numpy)
    taus = np.arange(min_tau, max_tau)
    diff = np.array([float(np.dot(trecho[:n-t]-trecho[t:], trecho[:n-t]-trecho[t:])) for t in taus])

    # CMND
    cumsum = np.cumsum(diff)
    cmnd   = diff * taus / (cumsum + 1e-10)

    # Primeiro minimo abaixo do threshold
    below = np.where(cmnd < 0.15)[0]
    if len(below) > 0:
        tau = taus[below[0]]
        return float(SR / tau) if tau > 0 else 0.0

    best = taus[int(np.argmin(cmnd))]
    return float(SR / best) if best > 0 else 0.0


def pitch_shift_scipy(seg, n_steps):
    """Pitch shift via resample — zero dependencias pesadas."""
    if abs(n_steps) < 0.05:
        return seg
    fator  = 2.0 ** (n_steps / 12.0)
    n_orig = len(seg)
    n_novo = int(round(n_orig / fator))
    if n_novo < 2:
        return seg
    shifted = scipy_resample(seg, n_novo).astype(np.float32)
    # Ajusta tamanho de volta ao original
    if len(shifted) >= n_orig:
        return shifted[:n_orig]
    else:
        return np.pad(shifted, (0, n_orig - len(shifted)))

def autotune(y, tonica, escala, strength=0.8):
    """Autotune frame a frame: chunks de 1s, yin proprio, scipy resample."""
    escala_midi = gerar_escala(tonica, escala)
    chunk_size  = SR   # 1 segundo
    n_chunks    = max(1, int(np.ceil(len(y) / chunk_size)))
    print(f'[autotune] {n_chunks} chunks')
    out = []

    for i in range(n_chunks):
        start = i * chunk_size
        end   = min(start + chunk_size, len(y))
        seg   = y[start:end].copy()

        freq = yin_simples(seg)
        if freq > 60:
            m_atual = freq_para_midi(freq)
            m_alvo  = nota_mais_proxima(m_atual, escala_midi)
            n_steps = (m_alvo - m_atual) * strength
            print(f'[autotune] chunk={i} freq={freq:.1f}Hz steps={n_steps:.2f}')
            if abs(n_steps) > 0.05:
                seg = pitch_shift_scipy(seg, n_steps)
        out.append(seg)

    return np.concatenate(out).astype(np.float32)

# ── Ruído ───────────────────────────────────────────────────────────────────
def eliminar_ruido(y, intensidade=0.5):
    ruido_rms = float(np.sqrt(np.mean(y[:SR//2] ** 2))) if len(y) > SR//2 else 0.01
    threshold = ruido_rms * (1.0 + intensidade * 3.0)
    rms_frame = np.sqrt(np.mean(y ** 2))
    if rms_frame < threshold:
        return y * max(0.0, 1.0 - intensidade)
    return y

# ── Rotas Flask ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processar', methods=['POST'])
def processar():
    arq = request.files.get('audio')
    if not arq:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    tonica        = request.form.get('tonica', 'C')
    escala        = request.form.get('escala', 'maior')
    strength      = float(request.form.get('strength', 0.8))
    reducao_ruido = float(request.form.get('reducao_ruido', 0.0))
    eq_graves     = float(request.form.get('eq_graves', 0.0))
    eq_medios     = float(request.form.get('eq_medios', 0.0))
    eq_agudos     = float(request.form.get('eq_agudos', 0.0))

    uid       = str(uuid.uuid4())
    nome_orig = arq.filename or ''
    ext_orig  = os.path.splitext(nome_orig)[1].lower()
    if not ext_orig:
        ct = (arq.content_type or '').lower()
        if 'mp3' in ct or 'mpeg' in ct:  ext_orig = '.mp3'
        elif 'mp4' in ct or 'm4a' in ct: ext_orig = '.mp4'
        elif 'ogg' in ct:                ext_orig = '.ogg'
        elif 'webm' in ct:               ext_orig = '.webm'
        elif 'wav' in ct:                ext_orig = '.wav'
        else:                            ext_orig = '.mp3'

    orig      = os.path.join(app.config['UPLOAD_FOLDER'], uid + ext_orig)
    tmp_files = []

    try:
        arq.save(orig); tmp_files.append(orig)
        print(f'[proc] {nome_orig} ext={ext_orig} size={os.path.getsize(orig)}')

        wav = converter_wav(orig); tmp_files.append(wav)
        y   = carregar_audio(wav)
        print(f'[proc] dur={len(y)/SR:.1f}s samples={len(y)}')

        if len(y) == 0:
            return jsonify({'erro': 'Audio sem conteudo.'}), 400

        if reducao_ruido > 0:
            y = eliminar_ruido(y, reducao_ruido)

        y = aplicar_eq(y, eq_graves, eq_medios, eq_agudos)
        y = autotune(y, tonica, escala, strength)

        # Normaliza volume
        fator = 10 ** (6.0 / 20.0)
        y = np.clip(y * fator, -1.0, 1.0).astype(np.float32)

        wav_out = os.path.join(app.config['PROCESSED_FOLDER'], f'tmp_{uid}.wav')
        mp3_out = os.path.join(app.config['PROCESSED_FOLDER'], f'venenno_{uid}.mp3')
        tmp_files.append(wav_out)

        sf.write(wav_out, y, SR)
        wav_para_mp3(wav_out, mp3_out)

        with open(mp3_out, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({'sucesso': True, 'audio_b64': audio_b64})

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
    return send_from_directory(app.config['PROCESSED_FOLDER'], nome, as_attachment=True)

@app.route('/debug')
def debug():
    return jsonify({'ffmpeg': FFMPEG, 'sr': SR})

@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response

import threading, time
_jobs = {}

@app.route('/enviar', methods=['POST'])
def enviar():
    dados = request.get_json(force=True)
    audio_b64     = dados.get('audio_b64', '') or dados.get('audio', '')
    tonica        = dados.get('tonica', 'C')
    escala        = dados.get('escala', 'maior')
    strength      = float(dados.get('strength', 0.8))
    reducao_ruido = float(dados.get('reducao_ruido', 0.0))
    eq_graves     = float(dados.get('eq_graves', 0.0))
    eq_medios     = float(dados.get('eq_medios', 0.0))
    eq_agudos     = float(dados.get('eq_agudos', 0.0))
    semitons_extra= int(dados.get('semitons_extra', 0))

    if not audio_b64:
        return jsonify({'erro': 'audio_b64 vazio'}), 400

    uid = str(uuid.uuid4())
    _jobs[uid] = {'status': 'processing'}

    def processar_bg():
        tmp_files = []
        try:
            raw = base64.b64decode(audio_b64)
            # Detecta formato pelo header
            if raw[:4] == b'fLaC':        ext = '.flac'
            elif raw[:3] == b'ID3' or raw[:2] == b'\xff\xfb': ext = '.mp3'
            elif raw[:4] == b'OggS':       ext = '.ogg'
            elif raw[:4] == b'RIFF':       ext = '.wav'
            elif raw[4:8] == b'ftyp':      ext = '.mp4'
            elif raw[:4] == b'\x1aE\xdf\xa3': ext = '.webm'
            else:                          ext = '.webm'

            orig = os.path.join(app.config['UPLOAD_FOLDER'], uid + ext)
            with open(orig, 'wb') as f:
                f.write(raw)
            tmp_files.append(orig)
            print(f'[enviar] uid={uid} ext={ext} size={len(raw)}')

            wav = converter_wav(orig); tmp_files.append(wav)
            y   = carregar_audio(wav)
            print(f'[enviar] dur={len(y)/SR:.1f}s')
            MAX_DUR = SR * 180  # maximo 3 minutos
            if len(y) > MAX_DUR:
                print(f'[enviar] audio muito longo, cortando em 3min')
                y = y[:MAX_DUR]

            if len(y) == 0:
                _jobs[uid] = {'status':'error','erro':'Audio sem conteudo.'}
                return

            if reducao_ruido > 0:
                y = eliminar_ruido(y, reducao_ruido)
            y = aplicar_eq(y, eq_graves, eq_medios, eq_agudos)
            y = autotune(y, tonica, escala, strength)

            # Semitons extra (manual)
            if semitons_extra != 0:
                y = pitch_shift_scipy(y, SR, semitons_extra)

            fator = 10 ** (6.0 / 20.0)
            y = np.clip(y * fator, -1.0, 1.0).astype(np.float32)

            wav_out = os.path.join(app.config['PROCESSED_FOLDER'], f'tmp_{uid}.wav')
            mp3_out = os.path.join(app.config['PROCESSED_FOLDER'], f'venenno_{uid}.mp3')
            tmp_files.append(wav_out)

            sf.write(wav_out, y, SR)
            wav_para_mp3(wav_out, mp3_out)

            with open(mp3_out, 'rb') as f:
                resultado_b64 = base64.b64encode(f.read()).decode('utf-8')

            _jobs[uid] = {'status':'done','audio_b64': resultado_b64}
            print(f'[enviar] uid={uid} concluido!')

        except Exception as e:
            print(f'[enviar] ERRO: {e}')
            import traceback; traceback.print_exc()
            _jobs[uid] = {'status':'error','erro': str(e)}
        finally:
            for f in tmp_files:
                if f and os.path.exists(f):
                    try: os.remove(f)
                    except: pass

    t = threading.Thread(target=processar_bg, daemon=True)
    t.start()
    return jsonify({'job_id': uid})


@app.route('/status/<job_id>')
def status(job_id):
    job = _jobs.get(job_id, {'status':'not_found'})
    # Limpa jobs antigos (>30)
    if len(_jobs) > 30:
        keys = list(_jobs.keys())
        for k in keys[:-20]:
            _jobs.pop(k, None)
    return jsonify(job)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
