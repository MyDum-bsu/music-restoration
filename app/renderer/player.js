// Player logic: switching variants in sync at the same playback position,
// blind labels, rating, saving to ratings.json via IPC.

const el = (id) => document.getElementById(id);
const NAME_RU = { original: "original", master: "master", ours: "ours" };

const state = {
  pack: null,          // pack.json + dir
  dir: null,
  index: 0,
  current: null,       // letter of the active variant
  ratings: {},
  loop: true,
  ready: false,
};

let audio = new Audio();
audio.preload = 'auto';

// --- loading a pack --------------------------------------------------------
async function openPack(dir) {
  const pack = await window.api.loadPack(dir);
  state.pack = pack;
  state.dir = dir;
  state.index = 0;
  state.ratings = (await window.api.loadRatings(dir)).ratings || {};
  el('packInfo').textContent = `${pack.name} · ${pack.n_items} items · target ${pack.target_lufs} LUFS` +
    (pack.has_ours ? ' · our variant included' : ' · no "ours" variant');
  console.log('[listen] pack loaded:', dir, pack.n_items, 'items');
  renderList();
  select(0);
}

// --- item list -------------------------------------------------------------
function renderList() {
  const ol = el('items');
  ol.innerHTML = '';
  state.pack.items.forEach((it, i) => {
    const li = document.createElement('li');
    li.id = `item-${i}`;
    const done = !!(state.ratings[it.id] && state.ratings[it.id].choice);
    if (done) li.classList.add('done');
    li.innerHTML = `<span>${i + 1}. ${it.id}</span><span>${it.artist || ''}</span>`;
    li.onclick = () => select(i);
    ol.appendChild(li);
  });
  updateProgress();
}

function updateProgress() {
  const done = state.pack.items.filter((it) => state.ratings[it.id] && state.ratings[it.id].choice).length;
  el('progress').textContent = `${done} / ${state.pack.items.length}`;
}

// --- selecting an item -----------------------------------------------------
function select(i) {
  state.index = i;
  const it = state.pack.items[i];
  el('itemTitle').textContent = `${i + 1}. ${it.id}`;
  el('itemNote').textContent = it.note || '';
  document.querySelectorAll('#items li').forEach((n) => n.classList.remove('active'));
  const li = el(`item-${i}`);
  if (li) li.classList.add('active');

  const labels = Object.keys(it.labels);
  const box = el('variants');
  box.innerHTML = '';
  labels.forEach((L) => {
    const f = it.files[L];
    const div = document.createElement('button');
    div.className = 'variant';
    div.id = `variant-${L}`;
    div.innerHTML = `<div class="key">variant ${L}</div><div class="name">${L}</div>
      <div class="meta">${f.lufs ?? '—'} LUFS · ${f.gain_db} dB</div>`;
    div.onclick = () => switchTo(L);
    box.appendChild(div);
  });

  const r = state.ratings[it.id] || {};
  const rb = el('ratingButtons');
  rb.innerHTML = '';
  labels.forEach((L) => {
    const b = document.createElement('button');
    b.textContent = `${L} is better`;
    b.onclick = () => rate(L);
    if (r.choice === L) b.classList.add('toggled');
    rb.appendChild(b);
  });
  const eq = document.createElement('button');
  eq.textContent = 'equal';
  eq.onclick = () => rate('equal');
  if (r.choice === 'equal') eq.classList.add('toggled');
  rb.appendChild(eq);

  state.ready = false;
  switchTo(labels[0], true);
  el('reveal').textContent = r.choice ? revealText(it) : '';
  el('levelNote').textContent =
    `All variants are loudness-matched to ${state.pack.target_lufs} LUFS — level differences will not give the answer away.`;
}

function revealText(it) {
  return 'Answer: ' + Object.keys(it.labels).map((L) => `${L} = ${NAME_RU[it.labels[L]] || it.labels[L]}`).join(', ');
}

// --- switching variants while keeping the position --------------------------
function switchTo(L, first = false) {
  console.log('[listen] variant', L, '→', state.pack.items[state.index].files[L].file);
  const it = state.pack.items[state.index];
  const wasPlaying = !audio.paused;
  const pos = first ? 0 : audio.currentTime;
  state.current = L;
  audio.src = `pack://pack/${it.files[L].file}`;
  document.querySelectorAll('.variant').forEach((n) => n.classList.remove('active'));
  const node = el(`variant-${L}`);
  if (node) node.classList.add('active');
  audio.addEventListener('loadedmetadata', function once() {
    audio.removeEventListener('loadedmetadata', once);
    if (!first) audio.currentTime = Math.min(pos, audio.duration || pos);
    state.ready = true;
    if (wasPlaying && !first) audio.play();
    updateTime();
  });
}

// --- transport -------------------------------------------------------------
function toggle() {
  if (audio.paused) { audio.play(); } else { audio.pause(); }
  el('btnPlay').textContent = audio.paused ? '▶ Play' : '❚❚ Pause';
}

function updateTime() {
  const d = audio.duration || 0;
  const c = audio.currentTime || 0;
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  el('time').textContent = `${fmt(c)} / ${fmt(d)}`;
  el('seek').value = d ? Math.round((c / d) * 1000) : 0;
}

audio.addEventListener('timeupdate', updateTime);
audio.addEventListener('play', () => { el('btnPlay').textContent = '❚❚ Pause'; });
audio.addEventListener('pause', () => { el('btnPlay').textContent = '▶ Play'; });
audio.addEventListener('ended', () => { if (state.loop && state.ready) { audio.currentTime = 0; audio.play(); } });

el('seek').addEventListener('input', (e) => {
  if (!state.ready) return;
  audio.currentTime = (e.target.value / 1000) * (audio.duration || 0);
});

// --- rating ----------------------------------------------------------------
async function rate(choice) {
  const it = state.pack.items[state.index];
  const payload = { labels: it.labels, choice, variant: state.current, key: it.key };
  state.ratings[it.id] = { ...payload };
  await window.api.saveRating(state.dir, it.id, payload);
  select(state.index);
  updateProgress();
  renderList();
}

// --- hotkeys ---------------------------------------------------------------
document.addEventListener('keydown', (e) => {
  if (!state.pack) return;
  const labels = Object.keys(state.pack.items[state.index].labels);
  if (e.code === 'Space') { e.preventDefault(); toggle(); }
  else if (e.key >= '1' && e.key <= '4') {
    const L = labels[Number(e.key) - 1];
    if (L) switchTo(L);
  }
  else if (e.key === 'ArrowLeft') { audio.currentTime = Math.max(0, audio.currentTime - 2); }
  else if (e.key === 'ArrowRight') { audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 2); }
  else if (e.key === 'l' || e.key === 'L') { state.loop = !state.loop; el('btnLoop').classList.toggle('toggled', state.loop); }
  else if (e.key === 'r' || e.key === 'R') {
    el('reveal').textContent = revealText(state.pack.items[state.index]);
  }
});

el('btnPlay').onclick = toggle;
el('btnLoop').onclick = () => { state.loop = !state.loop; el('btnLoop').classList.toggle('toggled', state.loop); };
el('btnPick').onclick = async () => { const d = await window.api.pickPack(); if (d) openPack(d); };
el('btnCsv').onclick = async () => { if (state.dir) { const p = await window.api.exportCsv(state.dir); el('packInfo').textContent = 'CSV: ' + p; } };
el('btnReveal').onclick = () => { el('reveal').textContent = revealText(state.pack.items[state.index]); };

window.api.onOpenPack((dir) => openPack(dir));
