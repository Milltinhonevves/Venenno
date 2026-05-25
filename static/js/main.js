'use strict';

let mediaRecorder = null;
let chunks = [];
let gravando = false;
let arquivoAtual = null;
let blobGravado = null;
let timerInterval = null;
let startTime = null;

function getMime() {
  const lista = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/ogg','audio/mp4'];
  for (const m of lista) {
    try { if (MediaRecorder.isTypeSupported(m)) return m; } catch(_) {}
  }
  return '';
}

function formatTime(ms) {
  const s = Math.floor(ms/1000);
  return String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
}

function setStatus(txt, cor) {
  const el = document.getElementById('status-rec');
  if (el) { el.textContent = txt; el.style.color = cor || ''; }
}

function mostrarErro(msg) {
  const el = document.getElementById('erro');
  if (el) { el.textContent = '❌ ' + msg; el.hidden = false; }
  const res = document.getElementById('resultado');
  if (res) res.hidden = true;
}

function mostrarResultado(url) {
  const el = document.getElementById('resultado');
  const dl = document.getElementById('btn-download');
  const erro = document.getElementById('erro');
  if (erro) erro.hidden = true;
  if (el) el.hidden = false;
  if (dl) { dl.href = url; dl.download = 'venenno.mp3'; }
  // Abre MP3 direto no player nativo do Android
  window.open(url, '_blank');
  const btnReusar = document.getElementById('btn-reusar');
  if (btnReusar && blobGravado) btnReusar.hidden = false;
}
  const btnReusar = document.getElementById('btn-reusar');
  if (btnReusar && blobGravado) btnReusar.hidden = false;
}

function setProcessando(sim) {
  const btn = document.getElementById('btn-processar');
  const txt = document.getElementById('btn-texto');
  const load = document.getElementById('btn-loading');
  if (btn) btn.disabled = sim;
  if (txt) txt.hidden = sim;
  if (load) load.hidden = !sim;
}

function resetGravador() {
  arquivoAtual = null;
  blobGravado = null;
  const previewBox = document.getElementById('preview-box');
  const previewAudio = document.getElementById('preview-audio');
  const btnProcessar = document.getElementById('btn-processar');
  if (previewBox) previewBox.hidden = true;
  if (previewAudio) previewAudio.src = '';
  if (btnProcessar) btnProcessar.disabled = true;
  document.getElementById('timer').textContent = '00:00';
  setStatus('Toque no microfone para gravar');
  ['btn-regravar','btn-apagar','btn-reusar'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.hidden = true;
  });
}

// Microfone
document.getElementById('btn-mic').addEventListener('click', async () => {
  if (gravando) {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = getMime();
    chunks = [];
    mediaRecorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const realMime = mediaRecorder.mimeType || mime || 'audio/webm';
      const ext = realMime.includes('ogg') ? 'ogg' : realMime.includes('mp4') ? 'mp4' : 'webm';
      blobGravado = new Blob(chunks, { type: realMime });
      arquivoAtual = new File([blobGravado], 'gravacao.' + ext, { type: realMime });
      stream.getTracks().forEach(t => t.stop());
      clearInterval(timerInterval);
      gravando = false;
      const btnMic = document.getElementById('btn-mic');
      btnMic.classList.remove('gravando');
      btnMic.textContent = '🎙️';
      const previewBox = document.getElementById('preview-box');
      const previewAudio = document.getElementById('preview-audio');
      const btnProcessar = document.getElementById('btn-processar');
      if (previewBox) previewBox.hidden = false;
      if (previewAudio) previewAudio.src = URL.createObjectURL(blobGravado);
      if (btnProcessar) btnProcessar.disabled = false;
      ['btn-regravar','btn-apagar'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.hidden = false;
      });
      setStatus('✅ Gravação pronta!', '#00ff88');
    };
    mediaRecorder.start(200);
    gravando = true;
    startTime = Date.now();
    const btnMic = document.getElementById('btn-mic');
    btnMic.classList.add('gravando');
    btnMic.textContent = '⏹️';
    timerInterval = setInterval(() => {
      document.getElementById('timer').textContent = formatTime(Date.now() - startTime);
    }, 500);
    setStatus('🔴 Gravando... toque para parar', '#ff3355');
  } catch (err) {
    mostrarErro('Microfone: ' + err.message);
  }
});

// Botões
const btnReg = document.getElementById('btn-regravar');
if (btnReg) btnReg.addEventListener('click', resetGravador);
const btnAp = document.getElementById('btn-apagar');
if (btnAp) btnAp.addEventListener('click', resetGravador);
const btnReusar = document.getElementById('btn-reusar');
if (btnReusar) btnReusar.addEventListener('click', () => {
  if (blobGravado) {
    arquivoAtual = new File([blobGravado], arquivoAtual.name, { type: arquivoAtual.type });
    btnReusar.hidden = true;
    document.getElementById('btn-processar').disabled = false;
    setStatus('✅ Pronto para processar de novo!', '#00ff88');
  }
});

// Selecionar arquivo
const audioInput = document.getElementById('audio-input');
if (audioInput) audioInput.addEventListener('change', () => {
  if (audioInput.files.length > 0) {
    arquivoAtual = audioInput.files[0];
    blobGravado = null;
    const previewBox = document.getElementById('preview-box');
    const previewAudio = document.getElementById('preview-audio');
    const btnProcessar = document.getElementById('btn-processar');
    if (previewBox) previewBox.hidden = false;
    if (previewAudio) previewAudio.src = URL.createObjectURL(arquivoAtual);
    if (btnProcessar) btnProcessar.disabled = false;
    setStatus('📂 ' + arquivoAtual.name, '#00ff88');
  }
});

// PROCESSAR — sem form submit, botão direto
document.getElementById('btn-processar').addEventListener('click', async (e) => {
  e.preventDefault();
  e.stopPropagation();

  if (!arquivoAtual) {
    mostrarErro('Grave ou selecione um áudio primeiro!');
    return;
  }

  document.getElementById('erro').hidden = true;
  document.getElementById('resultado').hidden = true;
  setProcessando(true);

  const fd = new FormData();
  fd.append('audio', arquivoAtual, arquivoAtual.name);
  fd.append('tonica',        document.querySelector('[name=tonica]').value);
  fd.append('escala',        document.querySelector('[name=escala]').value);
  fd.append('strength',      (parseInt(document.querySelector('[name=strength]').value) / 100).toFixed(2));
  fd.append('reducao_ruido', document.querySelector('[name=reducao_ruido]').value);
  fd.append('eq_graves',     document.querySelector('[name=eq_graves]').value);
  fd.append('eq_medios',     document.querySelector('[name=eq_medios]').value);
  fd.append('eq_agudos',     document.querySelector('[name=eq_agudos]').value);

  try {
    const resp = await fetch('/processar', { method: 'POST', body: fd });
    const texto = await resp.text();
    let data;
    try { data = JSON.parse(texto); } catch(_) {
      mostrarErro('Resposta inválida: ' + texto.substring(0,100));
      return;
    }
    if (data.sucesso) {
      mostrarResultado(window.location.origin + data.url);
    } else {
      mostrarErro(data.erro || 'Erro desconhecido');
    }
  } catch (err) {
    mostrarErro('Erro de rede: ' + err.message);
  } finally {
    setProcessando(false);
  }
});
