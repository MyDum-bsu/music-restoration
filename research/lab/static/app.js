/* Restoration Lab — plain JS, no build step. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const main = () => $("#main");

const AUTHOR = { "5618698": "Lisnoy Yozhyg", "3048819": "Alias Divas" };

/* Chart colours come from CSS so they follow the active theme. */
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}
const C = {
  get orig() { return cssVar("--orig"); },
  get master() { return cssVar("--master"); },
  get ours() { return cssVar("--ours"); },
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || "HTTP " + r.status);
  return j;
}

const n1 = v => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(1);
const n2 = v => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(2);
const sgn = (v, d = 1) => (v === null || v === undefined) ? "—"
  : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(Number(v)).toFixed(d);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ─────────────────────────────────────────────── charts */

function canvasBox(cv, h) {
  const host = cv.parentElement;
  const w = Math.max(Math.round(host ? host.clientWidth : 640), 300);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.width = w + "px";
  cv.style.height = h + "px";
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.font = '11px ui-monospace, Menlo, monospace';
  g.textBaseline = "middle";
  return [g, w, h];
}

function lineChart(cv, cfg) {
  const H = cfg.h || 220;
  const [g, W] = canvasBox(cv, H);
  const P = { l: 46, r: 12, t: 14, b: 30 };
  const series = cfg.series || [];
  const flat = a => a.flatMap(s => s.y.filter(v => Number.isFinite(v)));
  const ys = flat(series);
  if (!ys.length) return;

  const logX = !!cfg.logX;
  const lv = v => logX ? Math.log10(v) : v;
  const xs = series.flatMap(s => s.x.filter(v => Number.isFinite(v)));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = cfg.yMin !== undefined ? cfg.yMin : Math.min(...ys);
  let y1 = cfg.yMax !== undefined ? cfg.yMax : Math.max(...ys);
  if (y1 - y0 < 1e-9) { y1 += 1; y0 -= 1; }
  if (cfg.yMin === undefined) y0 -= (y1 - y0) * 0.08;
  if (cfg.yMax === undefined) y1 += (y1 - y0) * 0.08;

  const px = v => P.l + (lv(v) - lv(x0)) / (lv(x1) - lv(x0)) * (W - P.l - P.r);
  const py = v => H - P.b - (v - y0) / (y1 - y0) * (H - P.t - P.b);

  g.clearRect(0, 0, W, H);
  const cGrid = cssVar("--grid");
  const cText = cssVar("--ink-3");
  const cDash = cssVar("--rule-strong");
  g.strokeStyle = cGrid;
  g.fillStyle = cText;
  g.lineWidth = 1;

  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = y0 + (y1 - y0) * i / ticks;
    const y = Math.round(py(v)) + .5;
    g.beginPath(); g.moveTo(P.l, y); g.lineTo(W - P.r, y); g.stroke();
    g.textAlign = "right";
    g.fillText(v.toFixed(Math.abs(y1 - y0) < 4 ? 1 : 0), P.l - 8, y);
  }

  const xt = logX
    ? [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].filter(v => v >= x0 && v <= x1)
    : Array.from({ length: 5 }, (_, i) => x0 + (x1 - x0) * i / (5 - 1));
  g.textAlign = "center";
  for (const v of xt) {
    const x = Math.round(px(v)) + .5;
    g.strokeStyle = cGrid;
    g.beginPath(); g.moveTo(x, P.t); g.lineTo(x, H - P.b); g.stroke();
    g.fillStyle = cText;
    const lbl = cfg.xfmt ? cfg.xfmt(v) : (v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(Math.round(v)));
    g.fillText(lbl, x, H - P.b + 14);
  }

  if (cfg.hLine !== undefined) {
    g.strokeStyle = cDash; g.setLineDash([3, 4]);
    g.beginPath(); g.moveTo(P.l, py(cfg.hLine)); g.lineTo(W - P.r, py(cfg.hLine)); g.stroke();
    g.setLineDash([]);
  }

  for (const s of series) {
    g.strokeStyle = s.color;
    g.lineWidth = s.w || 1.5;
    if (s.dash) g.setLineDash([5, 3]);
    g.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const yv = s.y[i];
      if (!Number.isFinite(yv)) { pen = false; continue; }
      const X = px(s.x[i]), Y = py(yv);
      if (!pen) { g.moveTo(X, Y); pen = true; } else g.lineTo(X, Y);
    }
    g.stroke();
    g.setLineDash([]);
  }
}

function barList(el, rows) {
  const max = Math.max(...rows.map(r => r.v), 1e-9);
  el.innerHTML = rows.map(r => `
    <div class="bar">
      <span class="lbl">${esc(r.label)}</span>
      <span class="track"><span class="fill ${r.cls || ""}" style="width:${Math.max(r.v / max * 100, 1.5)}%"></span></span>
      <span class="val">${r.fmt ? r.fmt(r.v) : n2(r.v)}</span>
    </div>`).join("");
}

/* ─────────────────────────────────────────────── home */

async function viewHome() {
  const d = await api("/api/overview");
  const m = d.model_best || {};
  const ds = d.dataset || {};
  const ev = (d.audio_evals || []).slice(-1)[0] || null;
  const evs = ev && ev.variants ? ev.variants : {};

  // relative values: a person does not need to know what dB is
  // to grasp "closes 84% of the gap" and "lands within half a dB 38% of the time"
  const err = m.median_abs ?? null;
  const noop = m.baseline_noop ?? null;
  const gapClosed = (err != null && noop) ? Math.round((1 - err / noop) * 100) : null;
  const within = m["share_le_0.5"] ?? null;
  const withinPct = within != null ? Math.round(within * 100) : null;

  const approaches = [
    { label: "No processing", v: m.baseline_noop, note: "leave the album as released", cls: "bad" },
    { label: "Always add 4.6 dB", v: m.baseline_const46, note: "one fixed rule for everything" },
    { label: "Condition table", v: m.baseline_v0table, note: "three buckets by how crushed it is" },
    { label: "Model", v: m.median_abs, note: "predicts per album", cls: "hi" },
  ].filter(r => r.v != null);

  const maxErr = Math.max(...approaches.map(a => a.v));

  const classes = d.classes || {};
  const au = d.authors || {};

  main().innerHTML = `
  <h1>How closely does the algorithm match a mastering engineer?</h1>
  <p class="lede">Every album in this corpus exists twice: the loudness-war original,
  and a professional restoration of it. The pipeline predicts how much dynamic range
  that engineer gave back, then applies exactly that amount.</p>

  <div class="figures">
    <div class="fig">
      <div class="k">Recovers the engineer's intent</div>
      <div class="v">${gapClosed != null ? gapClosed + "%" : "—"}</div>
      <div class="n">of the distance between doing nothing and the engineer</div>
    </div>
    <div class="fig">
      <div class="k">Near-perfect calls</div>
      <div class="v">${withinPct != null ? withinPct + "%" : "—"}</div>
      <div class="n">of albums land within half a decibel of the engineer</div>
    </div>
    <div class="fig">
      <div class="k">Albums studied</div>
      <div class="v">${ds.pairs || "—"}</div>
      <div class="n">${ds.tracks || "—"} tracks from two independent engineers</div>
    </div>
    <div class="fig">
      <div class="k">Corpus</div>
      <div class="v">${n1(d.budget?.dataset_gb)}<span style="font-size:16px"> GB</span></div>
      <div class="n">${d.budget?.free_gb ?? "—"} GB free on the drive</div>
    </div>
  </div>

  <h2>Is that actually good?</h2>
  <div class="cols">
    <div>
      <div class="dotplot">
        ${approaches.map(a => `
          <div class="dot-row ${a.cls === "hi" ? "best" : ""}">
            <span class="dl">${esc(a.label)}</span>
            <span class="dtrack"><span class="ddot ${a.cls || ""}" style="left:${Math.max(2, Math.min(98, a.v / maxErr * 100))}%"></span></span>
            <span class="dv">${a.v === 0 ? "exact" : n2(a.v) + " dB"}</span>
          </div>`).join("")}
        <div class="dot-axis">
          <span></span>
          <span class="sp"><span>0 = matches</span><span>${n1(maxErr)} dB = far off</span></span>
          <span></span>
        </div>
      </div>
      <p class="chart-cap">Distance from the engineer's own decision, per album, averaged.
      “No processing” is what shipping the original sounds like to this measurement.</p>
    </div>
    <div>
      <h3>In plain terms</h3>
      <p class="note">A crushed album needs roughly four to five decibels of dynamic range
      given back. “Always add 4.6 dB” sounds close, but it is wrong by ${n2(m.baseline_const46)} dB
      on average, about a third of the whole job, because albums differ in how badly they were
      squashed and how much headroom the engineer chose to leave.</p>
      <p class="note" style="margin-top:14px">The model is wrong by ${n2(err)} dB. Put differently,
      it captures ${gapClosed}% of the ground between leaving the album alone and the engineer's
      decision. On ${withinPct}% of albums the miss is under half a decibel, which is below what
      most listeners can separate on a level-matched comparison.</p>
      <p class="note" style="margin-top:14px">The remaining error is honest: the engineer's taste
      is not fully predictable from the audio. The model is also held out by <em>album</em>, so
      no track from a test album was ever seen in training.</p>
    </div>
  </div>

  <h2>What is in the corpus</h2>
  <div class="cols even">
    <div>
      <div class="bars" id="bars-classes"></div>
      <p class="chart-cap">How damaged the original masters are, judged from their own dynamics.
      Most are heavily crushed, which is the population the tool exists for.</p>
    </div>
    <div>
      <div class="bars" id="bars-authors"></div>
      <p class="chart-cap">Two engineers with different taste. Neither one's albums are used to
      score the other, so the model cannot simply memorise one person's habits.</p>
    </div>
  </div>

  ${evs.model || evs.class ? `
  <h2>Check on real audio</h2>
  <div class="cols">
    <div>
      <div class="bars" id="bars-audio"></div>
      <p class="chart-cap">The same prediction task measured on short excerpts rather than whole
      albums. Deliberately small sample, so treat it as indicative rather than final.</p>
    </div>
    <div>
      <h3>Does it damage the signal?</h3>
      <p class="note">Matching the target is only half the job. The other half is not leaving
      marks: the difference between the original and our output is measured against the original
      itself, so anything the restoration invents shows up here.</p>
      ${(d.residual || []).map(r => `
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid var(--rule);padding:8px 0;font-size:13px">
          <span class="muted">${esc(r.name)}</span>
          <span class="num">${r.artifact_peaks === 0
            ? '<span style="color:var(--master)">clean</span>'
            : `${r.artifact_peaks} artefact${r.artifact_peaks === 1 ? "" : "s"}`}</span>
        </div>`).join("") || '<p class="note">No residual tests recorded.</p>'}
      <p class="chart-cap">Lower is better; “clean” means no artefact peaks were found above
      the detection threshold.</p>
    </div>
  </div>` : ""}
  `;

  barList($("#bars-classes"), [
    { label: "Heavily crushed", v: classes.crushed || 0, cls: "bad", fmt: v => String(v) },
    { label: "Moderate", v: classes.mid || 0, fmt: v => String(v) },
    { label: "Already healthy", v: classes.healthy || 0, cls: "hi", fmt: v => String(v) },
  ]);
  barList($("#bars-authors"), Object.entries(au).map(([pid, n]) => ({
    label: AUTHOR[pid] || pid, v: n, fmt: v => String(v),
  })));
  if ($("#bars-audio")) {
    barList($("#bars-audio"), [
      { label: "No processing", v: evs.noop?.median_abs, cls: "bad" },
      { label: "Condition table", v: evs.class?.median_abs },
      { label: "Model", v: evs.model?.median_abs, cls: "hi" },
    ].filter(r => r.v != null));
  }
}

/* ─────────────────────────────────────────────── dataset */

const sortState = { key: "artist", dir: 1 };
let pairRows = null;

async function viewDataset() {
  if (!pairRows) pairRows = (await api("/api/pairs")).rows || [];
  const rows = pairRows;

  const cls = ["crushed", "mid", "healthy"];
  main().innerHTML = `
  <h1>Corpus <span class="dim">${rows.length} album pairs</span></h1>
  <p class="lede">Every album where both a crushed original and a restored release exist.
  Open a pair to compare the two masters waveform by waveform.</p>

  <div class="filters">
    <input type="search" id="q" placeholder="Filter by artist, album or year">
    <select id="f-class"><option value="">All conditions</option>${cls.map(c => `<option>${c}</option>`).join("")}</select>
    <select id="f-author"><option value="">Both engineers</option><option value="5618698">Lisnoy Yozhyg</option><option value="3048819">Alias Divas</option></select>
  </div>

  <div class="tablewrap">
    <table id="tbl">
      <thead><tr>
        <th data-k="artist">Artist</th>
        <th data-k="album">Album</th>
        <th data-k="year" class="r">Year</th>
        <th data-k="src_class">Source</th>
        <th data-k="dr" class="r">DR orig → restored</th>
        <th data-k="dcrest" class="r">Δ crest</th>
        <th data-k="model_abs" class="r">Model error</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <p class="chart-cap">DR is TT-DR as measured on the whole album. Model error is the median
  prediction miss on that pair's excerpts.</p>
  `;

  $$("#tbl th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    sortState.dir = sortState.key === k ? -sortState.dir : 1;
    sortState.key = k;
    render();
  });

  function render() {
    const q = $("#q").value.toLowerCase().trim();
    const fc = $("#f-class").value;
    const fa = $("#f-author").value;
    const list = rows.filter(r => {
      if (fc && r.src_class !== fc) return false;
      if (fa && String(r.author_pid) !== fa) return false;
      return !q || `${r.artist} ${r.album} ${r.year}`.toLowerCase().includes(q);
    });
    const key = r => ({
      artist: (r.artist || "").toLowerCase(), album: (r.album || "").toLowerCase(),
      year: r.year || 0, src_class: r.src_class || "", dr: (r.dr_m - r.dr_o) || 0,
      dcrest: r.dcrest ?? -99, model_abs: r.model_abs ?? 99,
    }[sortState.key]);
    list.sort((a, b) => (key(a) > key(b) ? sortState.dir : -sortState.dir));

    $("#tbl tbody").innerHTML = list.map(r => `
      <tr data-key="${esc(r.key)}">
        <td class="title">${esc(r.artist)}<span class="sub">${AUTHOR[r.author_pid] || "unattributed"}</span></td>
        <td>${esc(r.album)}<span class="sub">${r.tracks || "—"} tracks</span></td>
        <td class="r">${r.year || "—"}</td>
        <td>${r.src_class === "healthy"
          ? `<span class="tag">${esc(r.src_class)}</span>`
          : `<span class="tag ${r.src_class === "crushed" ? "warn" : ""}">${esc(r.src_class || "?")}</span>`}</td>
        <td class="r"><span class="tag o">${n1(r.dr_o)}</span> → <span class="tag m">${n1(r.dr_m)}</span></td>
        <td class="r">${sgn(r.dcrest)}</td>
        <td class="r">${r.model_abs == null ? "—" : n2(r.model_abs)}</td>
      </tr>`).join("") || `<tr class="flat"><td colspan="7" class="empty-state">No albums match this filter.</td></tr>`;

    $$("#tbl tbody tr[data-key]").forEach(tr => tr.onclick = () => { location.hash = "#pair/" + tr.dataset.key; });
  }
  ["#q", "#f-class", "#f-author"].forEach(s => $(s).oninput = render);
  render();
}

/* ─────────────────────────────────────────────── pair */

async function viewPair(key) {
  const d = await api("/api/pair?key=" + encodeURIComponent(key));
  if (d.error) throw new Error(d.error);
  const rq = d.refq || {}, tr = d.tracks || [];
  const mean = f => tr.length ? tr.reduce((a, r) => a + f(r), 0) / tr.length : null;
  const dir = key.replace("/", "__");
  const rel = f => `dataset/pairs/${dir}/excerpts/${f}`;

  const groups = {};
  for (const e of d.excerpts || []) {
    const g = groups[`${e.track}|${e.tag}|${e.len_s}`] || (groups[`${e.track}|${e.tag}|${e.len_s}`] = {});
    g[e.who] = e;
  }
  const keys = Object.keys(groups).sort((a, b) => {
    const [t1, g1] = a.split("|"), [t2, g2] = b.split("|");
    return (+t1 - +t2) || g1.localeCompare(g2);
  });
  let ours = null;

  main().innerHTML = `
  <a class="back" href="#dataset">← All albums</a>
  <h1>${esc(rq.artist || d.meta.artist)} <span class="dim">${esc(rq.album || d.meta.album)}</span></h1>
  <p class="lede">${rq.year || d.meta.year || ""} · ${tr.length} tracks ·
  restored by ${AUTHOR[rq.author_pid] || "an unlabelled engineer"} ·
  <span class="tag ${rq.label === "trusted" ? "" : "warn"}">${esc(rq.label || "unlabelled")}</span></p>

  <div class="figures">
    <div class="fig">
      <div class="k">Dynamic range</div>
      <div class="v"><span style="color:var(--orig)">${n1(mean(r => r.original.dr_exact))}</span> → <span style="color:var(--master)">${n1(mean(r => r.restored.dr_exact))}</span></div>
      <div class="n">TT-DR, album mean</div>
    </div>
    <div class="fig">
      <div class="k">Crest restored</div>
      <div class="v">${sgn(mean(r => r.delta.crest))}<span style="font-size:16px"> dB</span></div>
      <div class="n">engineer's target</div>
    </div>
    <div class="fig">
      <div class="k">Level change</div>
      <div class="v">${sgn(mean(r => r.delta.rms))}<span style="font-size:16px"> dB</span></div>
      <div class="n">RMS, quieter is usual</div>
    </div>
    <div class="fig">
      <div class="k">Clipping added</div>
      <div class="v">${rq.added_clip ?? 0}</div>
      <div class="n">samples by the engineer</div>
    </div>
  </div>

  <h2>Compare an excerpt</h2>
  <div class="filters">
    <select id="pick" style="flex:1">
      ${keys.map(k => {
        const [t, tag, len] = k.split("|");
        const track = tr.find((_, i) => i + 1 === +t);
        return `<option value="${k}">Track ${t}${track && track.title ? " · " + esc(track.title) : ""} · ${tag} · ${len}s</option>`;
      }).join("") || '<option value="">No excerpts</option>'}
    </select>
    <button id="run">Run the pipeline</button>
  </div>
  <p id="run-note" class="chart-cap" style="min-height:18px"></p>

  <div class="split" style="margin-top:20px">
    <div>
      <div id="frames"></div>
      <div class="keys">
        <span><i style="background:${C.orig}"></i>Original</span>
        <span><i style="background:${C.master}"></i>Restored</span>
        <span id="key-ours" style="display:none"><i style="background:${C.ours}"></i>Ours</span>
      </div>
    </div>
    <div>
      <h3>Level over time</h3>
      <canvas id="cv-wave"></canvas>
      <p class="chart-cap">50 ms RMS windows. Flat line = no dynamics left.</p>
      <h3 style="margin-top:30px">Spectrum</h3>
      <canvas id="cv-spec"></canvas>
      <p class="chart-cap">Third-octave band levels. Overlapping curves mean the tonal
      balance survived restoration.</p>
      <div id="run-metrics" style="margin-top:28px"></div>
    </div>
  </div>

  <h2>All tracks</h2>
  <div class="tablewrap" style="max-height:none">
    <table>
      <thead><tr><th>#</th><th>Title</th><th class="r">DR</th><th class="r">Crest</th><th class="r">RMS</th><th class="r">Clipping</th></tr></thead>
      <tbody>
        ${tr.map((r, i) => `<tr class="flat">
          <td class="r" style="color:var(--ink-3)">${i + 1}</td>
          <td class="title">${esc(r.title)}</td>
          <td class="r"><span class="tag o">${r.original.dr}</span> → <span class="tag m">${r.restored.dr}</span></td>
          <td class="r">${n1(r.original.crest_db)} → ${n1(r.restored.crest_db)}</td>
          <td class="r">${n1(r.original.rms_db)} → ${n1(r.restored.rms_db)}</td>
          <td class="r">${(r.original.clip.L.clipped_samples + r.original.clip.R.clipped_samples) || 0} → ${(r.restored.clip.L.clipped_samples + r.restored.clip.R.clipped_samples) || 0}</td>
        </tr>`).join("")}
      </tbody>
    </table>
  </div>

  ${d.evals && d.evals.length ? `
  <h2>How the pipeline scored on this album</h2>
  <div class="tablewrap" style="max-height:none">
    <table>
      <thead><tr><th>Track</th><th>Segment</th><th class="r">Engineer</th><th class="r">Class table</th><th class="r">Model</th></tr></thead>
      <tbody>
      ${d.evals.map(r => `<tr class="flat">
        <td class="r">${r.track}</td>
        <td>${esc(r.tag)}</td>
        <td class="r">${n2(r.master_dcrest)}</td>
        <td class="r">${n2(r.class_dcrest)} <span class="muted">(${sgn(r.class_err, 2)})</span></td>
        <td class="r">${r.model_dcrest == null ? "—" : n2(r.model_dcrest) + ` <span class="muted">(${sgn(r.model_err, 2)})</span>`}</td>
      </tr>`).join("")}
      </tbody>
    </table>
  </div>
  <p class="chart-cap">Numbers are Δ crest in dB; the bracketed figure is the miss against the engineer.</p>` : ""}
  `;

  const BANDS = [22.4, 28.1, 35.5, 44.7, 56.1, 71.0, 89.4, 111.8, 141.4, 178.9, 223.6, 280.6,
    355.0, 447.2, 561.2, 709.9, 894.4, 1118.0, 1414.2, 1788.9, 2236.1, 2806.2,
    3549.6, 4472.1, 5612.5, 7099.3, 8944.3, 11180.3, 14142.1, 17888.5];

  if (tr.length) {
    const avg = side => BANDS.map((_, i) => tr.reduce((a, r) => a + r[side].bands_db[i], 0) / tr.length);
    lineChart($("#cv-wave"), {
      h: 200,
      series: [
        { x: BANDS, y: avg("original"), color: C.orig },
        { x: BANDS, y: avg("restored"), color: C.master },
      ],
      logX: true,
      xfmt: v => v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(Math.round(v)),
    });
  }

  async function drawFrames() {
    const k = $("#pick").value;
    if (!k) { $("#frames").innerHTML = '<p class="empty-state">No excerpts for this album.</p>'; return; }
    const g = groups[k];
    const [t, tag, len] = k.split("|");
    const pOrig = g.orig ? rel(g.orig.file) : null;
    const pMast = g.master ? rel(g.master.file) : null;

    const paths = [pOrig, ours ? ours.ours_path : null, pMast].filter(Boolean);
    // all three panels are drawn on one shared absolute dB scale — otherwise the comparison lies
    const specUrl = p => `/api/spectrogram?w=1100&h=300&path=${encodeURIComponent(p)}` +
      paths.filter(x => x !== p).map(x => `&path=${encodeURIComponent(x)}`).join("");

    const frame = (who, label, meta, path, empty) => `
      <div class="frame">
        <div class="frame-head">
          <span class="who ${who}">${label}</span>
          <span class="meta">${meta}</span>
        </div>
        ${path ? `<img src="${specUrl(path)}" alt="">
          <audio controls preload="none" src="/api/audio?path=${encodeURIComponent(path)}"></audio>`
          : `<div class="empty">${empty}</div>`}
      </div>`;

    $("#frames").innerHTML =
      frame("o", "Original", "as released", pOrig) +
      frame("our", "Ours", ours ? `${ours.class} · target ${sgn(ours.target_dcrest)} dB` : "not run yet",
        ours ? ours.ours_path : null, "Press “Run the pipeline” to render our version") +
      frame("m", "Restored", "by the engineer", pMast);

    const w1 = await api("/api/wave?path=" + encodeURIComponent(pOrig));
    const w2 = await api("/api/wave?path=" + encodeURIComponent(pMast));
    const series = [
      { x: w1.t, y: w1.rms, color: C.orig },
      { x: w2.t, y: w2.rms, color: C.master },
    ];
    if (ours) {
      $("#key-ours").style.display = "";
      const w3 = await api("/api/wave?path=" + encodeURIComponent(ours.ours_path));
      series.push({ x: w3.t, y: w3.rms, color: C.ours, dash: true });
    }
    lineChart($("#cv-wave"), { h: 200, series, xfmt: v => v.toFixed(1) + "s" });

    const s1 = await api("/api/spectrum?path=" + encodeURIComponent(pOrig));
    const s2 = await api("/api/spectrum?path=" + encodeURIComponent(pMast));
    const sSeries = [
      { x: s1.hz, y: s1.db, color: C.orig },
      { x: s2.hz, y: s2.db, color: C.master },
    ];
    if (ours) {
      const s3 = await api("/api/spectrum?path=" + encodeURIComponent(ours.ours_path));
      sSeries.push({ x: s3.hz, y: s3.db, color: C.ours, dash: true });
    }
    lineChart($("#cv-spec"), {
      h: 200, logX: true, series: sSeries,
      xfmt: v => v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(Math.round(v)),
    });
  }

  $("#run").onclick = async () => {
    const k = $("#pick").value;
    if (!k) return;
    const btn = $("#run");
    btn.disabled = true;
    $("#run-note").innerHTML = '<span class="spinner"></span>Processing excerpt…';
    try {
      ours = await api("/api/restore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: rel(groups[k].orig.file), ml: true }),
      });
      if (ours.error) throw new Error(ours.error);
      $("#run-note").textContent = `Done in ${ours.took_s}s · adaptive r = ${n2(ours.r_fit)}`;
      const ev = (d.evals || []).find(e => String(e.track) === k.split("|")[0]);
      $("#run-metrics").innerHTML = `
        <h3>Our pass versus the original</h3>
        <div class="mhead"><span></span><span>Original</span><span>Ours</span><span>Δ</span></div>
        <div class="mrows">
          ${mrow("TT-DR", ours.orig.dr, ours.ours.dr, sgn(ours.delta.dr, 0))}
          ${mrow("Crest, dB", n1(ours.orig.crest), n1(ours.ours.crest), sgn(ours.delta.crest))}
          ${mrow("RMS, dB", n1(ours.orig.rms), n1(ours.ours.rms), sgn(ours.ours.rms - ours.orig.rms))}
          ${mrow("True peak, dBFS", n1(ours.orig.tp), n1(ours.ours.tp), "")}
          ${mrow("Clipped samples", ours.orig.clip, ours.ours.clip, "")}
        </div>
        <p class="chart-cap">Material classified as <b>${ours.class}</b>, target
        ${sgn(ours.target_dcrest)} dB. ${ours.ml_target != null
          ? `The model asked for ${sgn(ours.ml_target)} dB (${esc(ours.ml_source)}).`
          : "The model was unavailable, so the class table set the target."}
        ${ev ? ` On this excerpt the model missed the engineer by ${sgn(ev.model_err, 2)} dB.` : ""}</p>`;
      await drawFrames();
    } catch (e) {
      $("#run-note").innerHTML = `<span class="err">${esc(e.message)}</span>`;
    }
    btn.disabled = false;
  };

  $("#pick").onchange = () => { ours = null; $("#run-metrics").innerHTML = ""; $("#key-ours").style.display = "none"; drawFrames(); };
  if (keys.length) drawFrames();
}

const mrow = (label, a, b, d) => `
  <div class="mrow"><span class="k">${label}</span><span class="a">${a}</span><span class="b">${b}</span>
  <span class="d">${d}</span></div>`;

/* ─────────────────────────────────────────────── models */

async function viewModels() {
  const d = await api("/api/models");
  const rt = d.runtime_model || {};
  const rows = d.rows || [];
  const album = rows.filter(r => r.holdout_mode === "by-album");
  const base = album[0] || {};

  main().innerHTML = `
  <h1>Which model sets the target</h1>
  <p class="lede">Each candidate predicts the crest-factor gain for a track it has never seen.
  A snapshot only reaches the pipeline if it beats the simpler alternatives on held-out albums.</p>

  <div class="figures">
    <div class="fig">
      <div class="k">In production</div>
      <div class="v" style="font-size:22px">${esc(rt.model || "—")}</div>
      <div class="n">${esc(rt.dataset_version || "")}</div>
    </div>
    <div class="fig">
      <div class="k">Its error</div>
      <div class="v">${n2(rt.median_abs)}<span style="font-size:16px"> dB</span></div>
      <div class="n">median, held-out albums</div>
    </div>
    <div class="fig">
      <div class="k">Training rows</div>
      <div class="v">${rt.n_train ?? "—"}</div>
      <div class="n">tracks with known target</div>
    </div>
    <div class="fig">
      <div class="k">Candidates tried</div>
      <div class="v">${d.n_snapshots ?? "—"}</div>
      <div class="n">snapshots in the registry</div>
    </div>
  </div>

  <h2>Accuracy by family</h2>
  <div class="cols">
    <div>
      <div class="bars" id="bars-lb"></div>
      <p class="chart-cap">Median absolute error, dB. Bars shown for models evaluated on held-out albums.</p>
    </div>
    <div>
      <h3>Why two holdout schemes</h3>
      <p class="note">Albums are held out <em>by album</em> to test whether the model generalises
      to new records. They are also held out <em>by engineer</em>, which is stricter: the model
      never sees that person's taste, so a good score there means we learned the task, not the hand.</p>
    </div>
  </div>

  <h2>Full leaderboard</h2>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>Model</th><th>Holdout</th><th class="r">Train / test</th>
        <th class="r">Median error</th><th class="r">Within 0.5 dB</th><th class="r">Bias</th><th>Gate</th>
      </tr></thead>
      <tbody>
      ${rows.map(r => `<tr class="flat">
        <td class="title">${esc(r.model)}${r.model_id === rt.id ? ' <span class="tag">live</span>' : ""}</td>
        <td>${r.holdout_mode === "by-album" ? "by album" : "by engineer"}</td>
        <td class="r">${r.n_train} / ${r.n_hold}</td>
        <td class="r">${n2(r.median_abs)}</td>
        <td class="r">${Math.round((r["share_le_0.5"] || 0) * 100)}%</td>
        <td class="r">${sgn(r.bias, 2)}</td>
        <td><span class="verdict ${r.promote ? "ok" : "no"}">${r.promote ? "pass" : "fail"}</span></td>
      </tr>`).join("")}
      </tbody>
    </table>
  </div>

  <h2>Baselines</h2>
  <div class="bars" id="bars-base"></div>
  <p class="chart-cap">Context for the table above: what the same task scores with no model at all.</p>
  `;

  barList($("#bars-lb"), album.map(r => ({
    label: r.model, v: r.median_abs, cls: r.promote ? "hi" : "",
  })));
  barList($("#bars-base"), [
    { label: "No processing", v: base.baseline_noop, cls: "bad" },
    { label: "Constant +4.6 dB", v: base.baseline_const46 },
    { label: "Class table (v0)", v: base.baseline_v0table },
  ].filter(r => r.v != null));
}

/* ─────────────────────────────────────────────── pipeline */

async function viewPipeline() {
  const d = await api("/api/policy");
  const p = d.defaults || {};
  const steps = [
    ["Classify", `Routes the track by measured dynamics: healthy albums pass untouched, quiet ones get a smaller target, dense ones the full lift.`, true],
    ["High-pass 25 Hz", "Removes rumble and DC offset with a zero-phase filter, so no transient is smeared.", p.hpf_hz],
    ["Declip", "Where flat-top clipping is found, rebuilds the missing peaks by autoregressive extrapolation. Nothing is invented that the signal does not support.", p.declip],
    ["Equalise", p.eq_match ? "Closed-loop match of band levels to the corpus target curve." : "Off by default; the expander already keeps the tonal balance.", p.eq_match],
    ["Expand contour", `The core stage. Per band and per frame, the level above the smoothed body is stretched (r ≈ ${p.r_expand}) while the body stays put. Adaptive pass tunes r until the crest gain hits the target.`, p.contour_enabled],
    ["Limit true peak", `Oversampled true-peak estimate, soft ceiling at ${p.target_tp_dbfs} dBFS, then 16-bit dither.`, true],
  ];

  main().innerHTML = `
  <h1>The chain</h1>
  <p class="lede">Six stages from a crushed master to a restored one. Every stage is bounded by a
  measurement, and anything that would exceed a bound is refused rather than shipped.</p>

  <div class="chain" style="margin-top:34px">
    ${steps.map(([name, desc], i) => `
      <div class="step">
        <div class="idx">0${i + 1}</div>
        <b>${esc(name)}</b>
        <p>${esc(desc)}</p>
      </div>`).join("")}
  </div>

  <h2>Gates</h2>
  <div class="cols">
    <div>
      <dl class="defs">
        <dt>True peak</dt><dd>≤ ${p.target_tp_dbfs} dBFS after the whole chain</dd>
        <dt>Clipping</dt><dd>zero samples added by the restoration</dd>
        <dt>Spectrum</dt><dd>band levels inside the reference tolerance</dd>
        <dt>Out of distribution</dt><dd>unfamiliar material falls back to the class table</dd>
        <dt>Model promotion</dt><dd>a snapshot must beat baselines on held-out albums</dd>
      </dl>
    </div>
    <div>
      <h3>What the gates are for</h3>
      <p class="note">Restoration is an exercise in not overreaching. A pass that raises crest factor
      by 8 dB is trivial to produce and destroys the record; the gates make the failure mode visible
      before a file is delivered rather than after someone listens to it.</p>
      <p class="note" style="margin-top:14px">The pipeline also refuses to guess: when the input does
      not resemble the corpus, it drops to a conservative class-based target instead of extrapolating.</p>
    </div>
  </div>

  <h2>Policy</h2>
  <p class="chart-cap" style="margin-bottom:14px">Live defaults read from the code. Changing one
  changes the next restoration, not this page.</p>
  <dl class="defs">
    ${Object.entries(p).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(JSON.stringify(v))}</dd>`).join("")}
  </dl>
  `;
}

/* ─────────────────────────────────────────────── router */

const views = { home: viewHome, dataset: viewDataset, models: viewModels, pipeline: viewPipeline };

async function route() {
  const raw = location.hash.slice(1) || "home";
  const parts = raw.split("/");
  const tab = parts[0];
  // pair keys contain a slash: "4916823/rel11"
  const arg = parts.length > 1 ? decodeURIComponent(parts.slice(1).join("/")) : "";

  $$("nav a").forEach(a => a.classList.toggle("active", a.dataset.tab === (tab === "pair" ? "dataset" : tab)));

  try {
    if (tab === "pair") {
      if (!/^\d+\/rel\d+$/.test(arg)) { location.hash = "#dataset"; return; }
      await viewPair(arg);
    } else if (views[tab]) {
      await views[tab]();
    } else {
      await views.home();
    }
  } catch (e) {
    main().innerHTML = `<h1>Something went wrong</h1><p class="err">${esc(e.message)}</p>
      <a class="back" href="#home">← Back to overview</a>`;
  }
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
if (document.readyState === "complete") route();
else window.addEventListener("load", route);

/* Theme: dark by default, remembered per browser. */
const THEME_KEY = "restore-lab-theme";
function applyTheme(name, redraw) {
  document.documentElement.dataset.theme = name;
  localStorage.setItem(THEME_KEY, name);
  $("#theme").textContent = name === "dark" ? "◐" : "◑";
  $("#theme").title = name === "dark" ? "Switch to light theme" : "Switch to dark theme";
  if (redraw) route();
}
applyTheme(localStorage.getItem(THEME_KEY) || "dark", false);
$("#theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next, true);
};

api("/api/overview").then(d => {
  const v = d.dataset?.version;
  $("#status").textContent = v ? `corpus ${v}` : "drive offline";
  $("#status").classList.toggle("bad", !v);
  $("#foot-note").textContent = `${d.dataset?.pairs ?? "—"} pairs · ${d.dataset?.tracks ?? "—"} tracks`;
}).catch(() => {
  $("#status").textContent = "server offline";
  $("#status").classList.add("bad");
});
