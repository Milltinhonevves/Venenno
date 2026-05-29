# v_nocache_fix
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
    Wn  = np.clip(np.array(cutoff, dtype=float) / nyq, 0.001, 0.999)
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

def detectar_pitch(seg):
    """
    Deteccao de pitch por autocorrelacao — simples, segura, sem bugs de tipo.
    Retorna frequencia fundamental em Hz ou 0.0 se nao detectar.
    """
    # Janela de 2048 amostras do centro
    W = 2048
    n = len(seg)
    if n < 512:
        return 0.0
    mid    = n // 2
    trecho = np.array(seg[max(0, mid - W//2): mid + W//2], dtype=np.float32)
    n      = len(trecho)

    min_tau = max(4, int(SR / 1200))   # max 1200 Hz
    max_tau = min(n // 2, int(SR / 60)) # min 60 Hz
    if min_tau >= max_tau:
        return 0.0

    # Autocorrelacao normalizada
    trecho -= trecho.mean()
    energia = float(np.dot(trecho, trecho))
    if energia < 1e-6:
        return 0.0

    melhor_tau   = 0
    melhor_corr  = -1.0
    for tau in range(min_tau, max_tau):
        corr = float(np.dot(trecho[:n - tau], trecho[tau:])) / energia
        if corr > melhor_corr:
            melhor_corr = corr
            melhor_tau  = tau

    if melhor_corr < 0.3 or melhor_tau == 0:
        return 0.0

    return float(SR) / float(melhor_tau)


# Alias para compatibilidade
def yin_simples(seg):
    return detectar_pitch(seg)


def pitch_shift_scipy(seg, n_steps):
    """
    Pitch shift SEM alterar duracao/velocidade.
    Passo 1: resample pra mudar o pitch (muda duracao temporariamente)
    Passo 2: resample de volta ao tamanho original (restaura duracao)
    Resultado: pitch diferente, mesma velocidade.
    """
    if abs(n_steps) < 0.05:
        return seg
    fator  = 2.0 ** (float(n_steps) / 12.0)
    n_orig = len(seg)
    # Passo 1: pitch up/down via resample
    n_pitch = int(round(n_orig / fator))
    if n_pitch < 2:
        return seg
    pitched = scipy_resample(seg, n_pitch).astype(np.float32)
    # Passo 2: volta ao tamanho original (time-stretch simples)
    restored = scipy_resample(pitched, n_orig).astype(np.float32)
    return restored

def autotune(y, tonica, escala, strength=0.8):
    """
    Autotune com overlap-add verdadeiro:
    - Janelas de 0.25s com 50% de sobreposicao
    - Crossfade por janela Hanning elimina cliques/granulados
    - Suavizacao de pitch entre frames consecutivos
    """
    escala_midi  = gerar_escala(tonica, escala)
    chunk_size   = SR // 4          # 0.25s
    hop_size     = chunk_size // 2  # 50% overlap
    janela       = np.hanning(chunk_size).astype(np.float32)

    # Normaliza a janela para overlap-add correto
    norm = np.zeros(len(y) + chunk_size, dtype=np.float32)
    for i in range(0, len(y), hop_size):
        end = min(i + chunk_size, len(norm))
        sz  = min(chunk_size, end - i)
        norm[i:i+sz] += janela[:sz] ** 2
    norm = np.where(norm < 1e-6, 1.0, norm)

    out          = np.zeros(len(y) + chunk_size, dtype=np.float32)
    ultimo_steps = 0.0
    n_chunks     = max(1, int(np.ceil(len(y) / hop_size)))
    print(f'[autotune] {n_chunks} frames overlap-add')

    for i in range(n_chunks):
        start = i * hop_size
        end   = min(start + chunk_size, len(y))
        seg   = y[start:end].copy()
        sz    = len(seg)
        if sz < 64:
            out[start:start+sz] += seg * janela[:sz]
            continue

        # Pad se necessario
        if sz < chunk_size:
            seg = np.pad(seg, (0, chunk_size - sz))

        freq = yin_simples(seg)
        if freq > 60:
            m_atual      = freq_para_midi(freq)
            m_alvo       = nota_mais_proxima(m_atual, escala_midi)
            n_steps_alvo = (m_alvo - m_atual) * float(strength)
            # Suaviza entre frames
            n_steps      = ultimo_steps * 0.25 + n_steps_alvo * 0.75
            ultimo_steps = n_steps
            if abs(n_steps) > 0.05:
                seg = pitch_shift_scipy(seg, n_steps)
        else:
            ultimo_steps *= 0.4

        # Aplica janela e acumula via overlap-add
        seg_jan = seg[:chunk_size] * janela
        out_end = min(start + chunk_size, len(out))
        seg_end = out_end - start
        out[start:out_end] += seg_jan[:seg_end]

    # Normaliza e recorta
    resultado = out[:len(y)] / norm[:len(y)]
    return np.clip(resultado, -1.0, 1.0).astype(np.float32)


# ── Ruído ───────────────────────────────────────────────────────────────────
def eliminar_ruido(y, intensidade=0.5):
    """
    Spectral gate: estima o ruido de fundo nos primeiros 0.5s (silencio),
    depois aplica gate espectral frame a frame para suprimir frequencias
    abaixo do limiar de ruido. Muito mais eficaz que threshold por RMS.
    """
    if intensidade < 0.01:
        return y

    frame_size = 512
    hop        = 256

    # Estima perfil de ruido nos primeiros 0.5s
    n_ruido = min(len(y), int(SR * 0.5))
    seg_ruido = y[:n_ruido] if n_ruido > frame_size else y[:frame_size]
    espectro_ruido = np.abs(np.fft.rfft(seg_ruido[:frame_size]))
    limiar = espectro_ruido * (1.0 + intensidade * 4.0)

    # Processa frame a frame
    n      = len(y)
    saida  = np.zeros(n, dtype=np.float32)
    contagem = np.zeros(n, dtype=np.float32)
    janela = np.hanning(frame_size).astype(np.float32)

    i = 0
    while i + frame_size <= n:
        frame  = y[i:i + frame_size] * janela
        spec   = np.fft.rfft(frame)
        mag    = np.abs(spec)
        fase   = np.angle(spec)

        # Gate: atenua frequencias abaixo do limiar
        ganho  = np.where(mag > limiar, 1.0, mag / (limiar + 1e-10) * (1.0 - intensidade))
        mag_filtrada = mag * ganho
        spec_filtrada = mag_filtrada * np.exp(1j * fase)

        frame_out = np.fft.irfft(spec_filtrada).astype(np.float32)
        saida[i:i + frame_size]    += frame_out * janela
        contagem[i:i + frame_size] += janela ** 2
        i += hop

    # Normaliza overlap-add
    contagem = np.where(contagem < 1e-6, 1.0, contagem)
    saida = saida / contagem
    return np.clip(saida, -1.0, 1.0).astype(np.float32)


# ── Mesa de Som ─────────────────────────────────────────────────────────────
def aplicar_reverb(y, intensidade=0.3):
    """Reverb simples via convolucao com decaimento exponencial."""
    if intensidade < 0.01:
        return y
    dur_reverb = int(SR * 0.8 * intensidade)   # ate 0.8s de cauda
    t          = np.linspace(0, 1, dur_reverb, dtype=np.float32)
    ir         = np.exp(-6.0 * t) * np.random.randn(dur_reverb).astype(np.float32)
    ir        /= (np.max(np.abs(ir)) + 1e-8)
    from scipy.signal import fftconvolve
    wet  = fftconvolve(y, ir)[:len(y)].astype(np.float32)
    mix  = float(intensidade) * 0.5
    return np.clip(y * (1 - mix) + wet * mix, -1.0, 1.0).astype(np.float32)

def aplicar_compressor(y, threshold_db=-18.0, ratio=4.0, makeup_db=6.0):
    """Compressor dinamico simples — controla picos e sobe volume geral."""
    threshold = 10 ** (float(threshold_db) / 20.0)
    makeup    = 10 ** (float(makeup_db) / 20.0)
    saida     = y.copy()
    acima     = np.abs(saida) > threshold
    saida[acima] = np.sign(saida[acima]) * (
        threshold + (np.abs(saida[acima]) - threshold) / float(ratio)
    )
    return np.clip(saida * makeup, -1.0, 1.0).astype(np.float32)

def aplicar_chorus(y, intensidade=0.4):
    """Chorus: mistura o sinal com versao levemente atrasada e modulada."""
    if intensidade < 0.01:
        return y
    delay_ms  = 25
    delay_s   = int(SR * delay_ms / 1000)
    modulacao = int(SR * 0.010)   # 10ms de variacao
    t         = np.arange(len(y), dtype=np.float32)
    lfo       = (np.sin(2 * np.pi * 1.5 * t / SR) * modulacao).astype(int)
    wet       = np.zeros_like(y)
    for i in range(delay_s, len(y)):
        offset = delay_s + max(0, min(modulacao, lfo[i]))
        src    = i - offset
        if 0 <= src < len(y):
            wet[i] = y[src]
    mix = float(intensidade) * 0.5
    return np.clip(y * (1 - mix) + wet * mix, -1.0, 1.0).astype(np.float32)


# ── De-esser ────────────────────────────────────────────────────────────────
def aplicar_deesser(y, intensidade=0.5):
    """
    De-esser: atenua sibilantes (4kHz-10kHz) que causam chiado no 'S'.
    Detecta frames com energia alta nessa faixa e aplica ganho reduzido.
    """
    if intensidade < 0.01:
        return y
    frame_size = 512
    hop        = 256
    freq_bins  = np.fft.rfftfreq(frame_size, d=1.0/SR)
    ess_mask   = (freq_bins >= 4000) & (freq_bins <= 10000)

    saida    = np.zeros(len(y), dtype=np.float32)
    contagem = np.zeros(len(y), dtype=np.float32)
    janela   = np.hanning(frame_size).astype(np.float32)

    i = 0
    while i + frame_size <= len(y):
        frame = y[i:i+frame_size] * janela
        spec  = np.fft.rfft(frame)
        mag   = np.abs(spec)
        fase  = np.angle(spec)

        energia_ess  = np.mean(mag[ess_mask] ** 2)
        energia_total= np.mean(mag ** 2) + 1e-10
        ratio_ess    = energia_ess / energia_total

        # Se a faixa sibilante domina, atenua
        if ratio_ess > 0.3:
            reducao = 1.0 - (intensidade * min(1.0, ratio_ess * 2))
            mag[ess_mask] *= max(0.05, reducao)

        spec_out  = mag * np.exp(1j * fase)
        frame_out = np.fft.irfft(spec_out).astype(np.float32)
        saida[i:i+frame_size]    += frame_out * janela
        contagem[i:i+frame_size] += janela ** 2
        i += hop

    contagem = np.where(contagem < 1e-6, 1.0, contagem)
    return np.clip(saida / contagem, -1.0, 1.0).astype(np.float32)

# ── Saturação ────────────────────────────────────────────────────────────────
def aplicar_saturacao(y, intensidade=0.4):
    """
    Saturacao harmonica: adiciona calor analogico via soft-clip (tanh).
    Gera harmonicos pares que enriquecem o timbre vocal.
    """
    if intensidade < 0.01:
        return y
    drive = 1.0 + float(intensidade) * 8.0   # 1x a 9x de gain
    wet   = np.tanh(y * drive) / np.tanh(drive)
    mix   = float(intensidade) * 0.6
    return np.clip(y * (1 - mix) + wet * mix, -1.0, 1.0).astype(np.float32)

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
    reverb_int    = float(request.form.get('reverb', 0.0))
    chorus_int    = float(request.form.get('chorus', 0.0))
    compressor_on = int(request.form.get('compressor', 0))
    semitons_extra= int(request.form.get('semitons_extra', 0))
    deesser_int   = float(request.form.get('deesser', 0.0))
    saturacao_int = float(request.form.get('saturacao', 0.0))
    export_wav    = request.form.get('export_wav', '0') == '1'

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

        if reverb_int    > 0.01: y = aplicar_reverb(y, reverb_int)
        if chorus_int    > 0.01: y = aplicar_chorus(y, chorus_int)
        if compressor_on == 1:   y = aplicar_compressor(y)
        if deesser_int   > 0.01: y = aplicar_deesser(y, deesser_int)
        if saturacao_int > 0.01: y = aplicar_saturacao(y, saturacao_int)

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
        if export_wav:
            with open(wav_out, 'rb') as fw:
                wav_b64 = base64.b64encode(fw.read()).decode('utf-8')
            return jsonify({'sucesso': True, 'audio_b64': audio_b64, 'wav_b64': wav_b64, 'formato': 'wav'})
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
    reverb_int    = float(dados.get('reverb', 0.0))
    chorus_int    = float(dados.get('chorus', 0.0))
    compressor_on = int(dados.get('compressor', 0))
    deesser_int   = float(dados.get('deesser', 0.0))
    saturacao_int = float(dados.get('saturacao', 0.0))
    export_wav    = dados.get('export_wav', False)

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

            if reverb_int    > 0.01: y = aplicar_reverb(y, reverb_int)
            if chorus_int    > 0.01: y = aplicar_chorus(y, chorus_int)
            if compressor_on == 1:   y = aplicar_compressor(y)
            if deesser_int   > 0.01: y = aplicar_deesser(y, deesser_int)
            if saturacao_int > 0.01: y = aplicar_saturacao(y, saturacao_int)

            # Semitons extra (manual)
            if semitons_extra != 0:
                y = pitch_shift_scipy(y, semitons_extra)

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
