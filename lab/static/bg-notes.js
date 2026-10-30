/* Фоновая анимация: контурные ноты, которые сами себя прорисовывают и стирают.
 *
 * Штрих не плоский: линия рисуется полой — как контур фигуры с прозрачной
 * заливкой. Внутри полосы вырезана середина маской, поэтому просвет
 * действительно прозрачен, а не закрашен цветом фона: сквозь него видно и фон,
 * и другие ноты. На выходе — две параллельные линии с постоянным просветом.
 *
 * Цикл одной ноты:
 *   draw  — полоса прорисовывается от начала к концу
 *   hold  — фигура стоит целиком, её видно
 *   erase — стирается от конца к началу с той же скоростью
 *
 * Рисование и стирание — один и тот же strokeDashoffset в разные стороны
 * (L -> 0, пауза, 0 -> L), поэтому скорости совпадают точно.
 */
(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("bg-notes");
  if (!svg) return;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const SHAPES = [
    // восьмая
    "M6 76a13 10 0 1 0 26 0a13 10 0 1 0 -26 0M32 76V10c13 4 19 15 14 31",
    // две восьмых с общей перекладиной
    "M10 80a11 8.5 0 1 0 22 0a11 8.5 0 1 0 -22 0" +
      "M48 70a11 8.5 0 1 0 22 0a11 8.5 0 1 0 -22 0" +
      "M32 80V14L70 6M70 70V6",
    // диез
    "M38 12V88M58 8V84M16 38H80M14 64H78",
    // бемоль
    "M34 8V86c0-15 28-17 28-31 0-10-9-15-17-11",
    // целая
    "M28 50a22 15 0 1 0 44 0a22 15 0 1 0 -44 0",
    // половинная
    "M26 62a13 9.5 0 1 0 26 0a13 9.5 0 1 0 -26 0M52 62V10",
  ];

  const RIBBON = 6.4;   // полная ширина полосы, px
  const CUT = 4.0;      // что вырезаем из середины -> ребро (6.4 - 4.0) / 2 = 1.2 px

  const PHASE = { draw: 4600, hold: 2400, erase: 4000 };
  const TOTAL = PHASE.draw + PHASE.hold + PHASE.erase;

  const rand = (a, b) => a + Math.random() * (b - a);
  const pick = a => a[Math.floor(Math.random() * a.length)];

  const defs = document.createElementNS(NS, "defs");
  svg.appendChild(defs);

  const notes = [];
  const COUNT = innerWidth < 760 ? 3 : 5;

  for (let i = 0; i < COUNT; i++) {
    const maskId = `bg-note-cut-${i}`;

    // маска: белое = видно, чёрная линия по той же траектории = прозрачный просвет
    const mask = document.createElementNS(NS, "mask");
    mask.setAttribute("id", maskId);
    mask.setAttribute("maskUnits", "userSpaceOnUse");
    mask.setAttribute("x", "0");
    mask.setAttribute("y", "0");
    mask.setAttribute("width", "100%");
    mask.setAttribute("height", "100%");

    const full = document.createElementNS(NS, "rect");
    full.setAttribute("x", "0");
    full.setAttribute("y", "0");
    full.setAttribute("width", "100%");
    full.setAttribute("height", "100%");
    full.setAttribute("fill", "#fff");
    mask.appendChild(full);

    const holder = document.createElementNS(NS, "g");
    const cutter = document.createElementNS(NS, "path");
    cutter.setAttribute("class", "cut");
    cutter.setAttribute("d", SHAPES[0]);
    cutter.setAttribute("stroke-width", String(CUT));
    holder.appendChild(cutter);
    mask.appendChild(holder);
    defs.appendChild(mask);

    const g = document.createElementNS(NS, "g");
    g.setAttribute("mask", `url(#${maskId})`);
    const trace = document.createElementNS(NS, "path");
    trace.setAttribute("class", "trace");
    trace.setAttribute("d", SHAPES[0]);
    trace.setAttribute("stroke-width", String(RIBBON));
    g.appendChild(trace);
    svg.appendChild(g);

    notes.push({ g, trace, holder, cutter });
  }

  const headerBottom = () => {
    const h = document.querySelector("header");
    return h ? h.getBoundingClientRect().bottom : 90;
  };

  function transformFor(size) {
    const vw = innerWidth, vh = innerHeight;
    const top = headerBottom() + 30;
    const x = rand(-0.08, 1.08) * vw;
    const y = rand(top, Math.max(top + 80, vh * 1.02));
    const rot = rand(-30, 30);
    const s = size / 100;              // viewBox фигур — условные 100 единиц
    return `translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${rot.toFixed(1)}) ` +
      `scale(${s.toFixed(3)}) translate(-50 -50)`;
  }

  function cycle(n) {
    const d = pick(SHAPES);
    n.trace.setAttribute("d", d);
    n.cutter.setAttribute("d", d);      // вырез идёт по той же траектории

    const minS = 70;
    const maxS = Math.max(300, Math.min(innerWidth, innerHeight) * 0.85);
    const size = minS + (maxS - minS) * Math.pow(Math.random(), 1.6);

    const t = transformFor(size);
    n.g.setAttribute("transform", t);
    n.holder.setAttribute("transform", t);

    const L = n.trace.getTotalLength();
    if (!L || !isFinite(L)) return;

    // полоса: L -> 0 (рисуем) -> пауза -> 0 -> L (стираем от конца)
    n.trace.style.strokeDasharray = `${L} ${L}`;
    n.trace.animate([
      { offset: 0, strokeDashoffset: L },
      { offset: PHASE.draw / TOTAL, strokeDashoffset: 0 },
      { offset: (PHASE.draw + PHASE.hold) / TOTAL, strokeDashoffset: 0 },
      { offset: 1, strokeDashoffset: L },
    ], { duration: TOTAL, easing: "linear", fill: "forwards" });

    n.timer = setTimeout(cycle, TOTAL + rand(0, 1200));
  }

  notes.forEach((n, i) => setTimeout(() => cycle(n), i * 900 + rand(0, 700)));

  let rt = null;
  addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => {
      for (const n of notes) {
        clearTimeout(n.timer);
        n.g.getAnimations({ subtree: true }).forEach(a => a.cancel());
        cycle(n);
      }
    }, 400);
  });
})();
