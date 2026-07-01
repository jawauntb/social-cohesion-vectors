import { TARGETS, type EngagementPrediction } from "@/lib/types";

// Normalized predictions arrive on ~0..1 (sigmoid) scale from the serve stub;
// clamp defensively so the bars render sanely for any model version.
export function EngagementBars({ pred }: { pred: EngagementPrediction }) {
  return (
    <div className="bars">
      {TARGETS.map((t) => {
        const raw = (pred[t] ?? 0) as number;
        const pct = Math.max(0, Math.min(1, raw)) * 100;
        return (
          <div className="bar-row" key={t}>
            <span className="label">{t}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${pct}%` }} />
            </span>
            <span className="val">{raw.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}
