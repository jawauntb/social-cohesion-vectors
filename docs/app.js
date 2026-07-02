/* Social Cohesion Vectors — background field + scroll reveals.
 * The field is a small visual metaphor: points drift, and a shared
 * "genuine" direction gently aligns them, while noise scatters others —
 * one manifold plus residual variation, not a literal depiction of data. */

(function () {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- scroll reveal ---- */
  const revealTargets = document.querySelectorAll(
    "section h2, .section-sub, .q-card, .idea-note, .finding, .lane, .fault-list li, .honesty-cols > div, .reproduce pre, .research-question blockquote"
  );
  revealTargets.forEach((el) => el.classList.add("reveal"));

  if ("IntersectionObserver" in window && !reduceMotion) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("in");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    revealTargets.forEach((el) => io.observe(el));
  } else {
    revealTargets.forEach((el) => el.classList.add("in"));
  }

  /* ---- background vector field ---- */
  const canvas = document.getElementById("field");
  if (!canvas || reduceMotion) return;
  const ctx = canvas.getContext("2d");

  let w, h, dpr, points;
  const GENUINE = { r: 95, g: 212, b: 168 };
  const PSEUDO = { r: 224, g: 138, b: 122 };
  const ACCENT = { r: 138, g: 166, b: 255 };

  // a fixed shared "genuine" direction the cohesive points align to
  const dir = { x: 0.82, y: -0.57 };

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
  }

  function seed() {
    const count = Math.round((w * h) / 16000);
    points = [];
    for (let i = 0; i < count; i++) {
      const genuine = Math.random() > 0.45;
      points.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        genuine,
        a: 0.15 + Math.random() * 0.4,
        len: 6 + Math.random() * 14,
      });
    }
  }

  let t = 0;
  function frame() {
    t += 0.004;
    ctx.clearRect(0, 0, w, h);

    for (const p of points) {
      if (p.genuine) {
        // pull velocity gently toward the shared direction
        p.vx += dir.x * 0.012;
        p.vy += dir.y * 0.012;
      } else {
        // scatter with slow rotating noise
        p.vx += Math.cos(t * 3 + p.x * 0.01) * 0.018;
        p.vy += Math.sin(t * 3 + p.y * 0.01) * 0.018;
      }
      // damping
      p.vx *= 0.94;
      p.vy *= 0.94;
      p.x += p.vx;
      p.y += p.vy;

      // wrap
      if (p.x < -20) p.x = w + 20;
      if (p.x > w + 20) p.x = -20;
      if (p.y < -20) p.y = h + 20;
      if (p.y > h + 20) p.y = -20;

      const sp = Math.hypot(p.vx, p.vy) || 0.0001;
      const nx = p.vx / sp;
      const ny = p.vy / sp;
      const c = p.genuine ? GENUINE : PSEUDO;

      ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${p.a})`;
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + nx * p.len, p.y + ny * p.len);
      ctx.stroke();

      ctx.fillStyle = `rgba(${ACCENT.r},${ACCENT.g},${ACCENT.b},${p.a * 0.5})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(frame);
  }

  window.addEventListener("resize", resize, { passive: true });
  resize();
  requestAnimationFrame(frame);
})();
