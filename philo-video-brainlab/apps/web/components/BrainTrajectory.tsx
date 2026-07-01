// Tiny dependency-free SVG sparkline of a per-step trajectory signal
// (e.g. velocity or surprise) — "how the brain moves", not where it is.
export function BrainTrajectory({
  series,
  label,
}: {
  series: number[];
  label?: string;
}) {
  const W = 600;
  const H = 160;
  const pad = 10;
  if (!series || series.length < 2) {
    return <div className="muted">No trajectory yet.</div>;
  }
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = max - min || 1;
  const step = (W - pad * 2) / (series.length - 1);
  const pts = series
    .map((v, i) => {
      const x = pad + i * step;
      const y = H - pad - ((v - min) / span) * (H - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <div>
      {label && <div className="muted" style={{ marginBottom: 6 }}>{label}</div>}
      <svg className="traj" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <polyline
          points={pts}
          fill="none"
          stroke="var(--brand)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}
