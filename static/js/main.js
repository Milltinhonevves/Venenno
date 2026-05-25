/* Venenno — main.js v1 */
'use strict';

// ── Elementos ──────────────────────────────────────────────────────────────
const btnMic       = document.getElementById('btn-mic');
const timerEl      = document.getElementById('timer');
const statusEl     = document.getElementById('status-rec');
const previewBox   = document.getElementById('preview-box');
const previewAudio = document.getElementById('preview-audio');
const btnRegravar  = document.getElementById('btn-regravar');
const btnApagar    = document.getElementById('btn-apagar');
const btnReusar    = document.getElementById('btn-reusar');
const audioInput   = document.getElementById('audio-input');
const btnProcessar = document.getElementById('btn-processar');
const formEl       = document.getElementById('form-venenno');
const resultadoEl  = document.getElementById('resultado');
const playerEl     = document.getElementById('player');
const downloadEl   = document.getElementById('btn-download');
const erroEl       = document.getElementById('erro');
const btnTexto     = document.getElementById('btn-texto');
const btnLoading   = document.getElementById('btn-loading');

// ── Estado ─────────────────────────────────────────────────────────────────
let mediaRecorder = null;
let chunks        = [];
let timerInterval = null;
let startTime     = null;
let gravando      = false;
let arquivoAtual  = null;
let blobGravado   = null;

// ── Mime ───────────────────────────────────────────────────────────────────
function getMime() {
  const lista = [
    'audio/webm;codecs=opus', 'audio/webm',
    'audio/ogg;codecs=opus',  'audio/ogg',
    'audio/mp4',
  ];
  for (const m of lista) {
    try { if (MediaRecorder.isTypeSupported(m)) return m; } catch (_) {}
  }
  return '';
}

function mimeParaExt(mime) {
  if (mime.includes('ogg')) return 'ogg';
  if (mime.includes('mp4')) return 'mp4';
  return 'webm';
}

// ── Timer ──────────────────────────────────────────────────────────────────
function formatTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function setStatus(txt, cor = '') {
  statusEl.textContent = txt;
  statusEl.style.color = cor || '';
}

function resetGravador() {
  arquivoAtual  = null;
  blobGravado   = null;
  previewBox.hidden     = true;
  previewAudio.src      = '';
  btnRegravar.hidden    = true;
  btnApagar.hidden      = true;
  btnReusar.hidden      = true;
  btnProcessar.disabled = true;
  timerEl.textContent   = '00:00';
  setStatus('Toque no microfone para gravar');
}

function mostrarPreview(blob, mime) {
  previewBox.hidden     = false;
  previewAudio.src      = URL.createObjectURL(blob);
  btnRegravar.hidden    = false;
  btnApagar.hidden      = false;
  btnReusar.hidden      = true;
  btnProcessar.disabled = false;
  setStatus('✅ Gravação pronta!', '#00ff88');
}

function setProcessando(sim) {
  btnProcessar.disabled = sim;
  btnTexto.hidden       = sim;
  btnLoading.hidden     = !sim;
}

function mostrarErro(msg) {
  erroEl.textContent = '❌ ' + msg;
  erroEl.hidden      = false;
  resultadoEl.hidden = true;
}

function mostrarResultado(url) {
  erroEl.hidden        = false; // vamos esconder
  erroEl.hidden        = true;
  resultadoEl.hidden   = false;
  playerEl.src         = url + '?t=' + Date.now();
  downloadEl.href      = url;
  if (blobGravado) btnReusar.hidden = false;
}

// ── Gravar / Parar ─────────────────────────────────────────────────────────
btnMic.addEventListener('click', async () => {
  if (gravando) {
    // Para gravação
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    return;
  }

  try {
    const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime     = getMime();
    chunks         = [];
    mediaRecorder  = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);

    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };

    mediaRecorder.onstop = () => {
      const realMime = mediaRecorder.mimeType || mime || 'audio/webm';
      const ext      = mimeParaExt(realMime);
      blobGravado    = new Blob(chunks, { type: realMime });
      arquivoAtual   = new File([blobGravado], `gravacao.${ext}`, { type: realMime });
      stream.getTracks().forEach(t => t.stop());
      clearInterval(timerInterval);
      gravando = false;
      btnMic.classList.remove('gravando');
      btnMic.textContent = '🎙️';
      mostrarPreview(blobGravado, realMime);
    };

    mediaRecorder.start(200);
    gravando  = true;
    startTime = Date.now();
    btnMic.classList.add('gravando');
    btnMic.textContent = '⏹️';
    timerInterval = setInterval(() => {
      timerEl.textContent = formatTime(Date.now() - startTime);
    }, 500);
    setStatus('🔴 Gravando... toque para parar', '#ff3355');

  } catch (err) {
    mostrarErro('Não foi possível acessar o microfone: ' + err.message);
  }
});

// ── Botões auxiliares ──────────────────────────────────────────────────────
btnRegravar.addEventListener('click', resetGravador);
btnApagar.addEventListener('click',   resetGravador);

btnReusar.addEventListener('click', () => {
  if (blobGravado) {
    arquivoAtual      = new File([blobGravado], arquivoAtual.name, { type: arquivoAtual.type });
    btnReusar.hidden  = true;
    btnProcessar.disabled = false;
    setStatus('✅ Pronto para processar de novo!', '#00ff88');
  }
});

audioInput.addEventListener('change', () => {
  if (audioInput.files.length > 0) {
    arquivoAtual          = audioInput.files[0];
    blobGravado           = null;
    previewBox.hidden     = false;
    previewAudio.src      = URL.createObjectURL(arquivoAtual);
    btnProcessar.disabled = false;
    btnRegravar.hidden    = true;
    btnApagar.hidden      = true;
    btnReusar.hidden      = true;
    setStatus('📂 ' + arquivoAtual.name, '#00ff88');
  }
});

// ── Processar ──────────────────────────────────────────────────────────────
formEl.addEventListener('submit', async e => {
  e.preventDefault();

  if (!arquivoAtual) {
    mostrarErro('Grave ou selecione um áudio primeiro!');
    return;
  }

  erroEl.hidden      = true;
  resultadoEl.hidden = true;
  setProcessando(true);

  const fd = new FormData(formEl);
  fd.set('audio', arquivoAtual, arquivoAtual.name);

  try {
    const resp = await fetch('/processar', { method: 'POST', body: fd });
    let texto  = '';
    let json   = null;

    try {
      texto = await resp.text();
      json  = JSON.parse(texto);
    } catch (_) {
      mostrarErro(`Servidor retornou HTTP ${resp.status}: ${texto.substring(0, 200)}`);
      return;
    }

    if (json && json.sucesso) {
      mostrarResultado(json.url);
    } else {
      mostrarErro(json && json.erro ? json.erro : `HTTP ${resp.status}: ${texto.substring(0, 200)}`);
    }

  } catch (err) {
    mostrarErro('Erro de conexão: ' + err.message);
  } finally {
    setProcessando(false);
  }
});
