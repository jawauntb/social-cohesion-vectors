"use client";

import { useState } from "react";
import { EngagementBars } from "@/components/EngagementBars";
import type { EngagementPrediction } from "@/lib/types";

export default function PredictPage() {
  const [videoId, setVideoId] = useState("draft-001");
  const [url, setUrl] = useState("");
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pred, setPred] = useState<EngagementPrediction | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPred(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId, url: url || null, caption: caption || null }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      setPred(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Pre-publication scoring</h1>
      <p className="lede">
        Estimate engagement <em>before</em> posting. The backend extracts a TRIBE v2 brain
        trajectory + multimodal features and applies the trained engagement models, then surfaces
        editor notes from trajectory landmarks.
      </p>

      <form onSubmit={onSubmit}>
        <label>
          Video id
          <input value={videoId} onChange={(e) => setVideoId(e.target.value)} required />
        </label>
        <label>
          Source URL (optional)
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://… or leave blank to use a mounted path"
          />
        </label>
        <label>
          Caption / hook (optional)
          <textarea value={caption} onChange={(e) => setCaption(e.target.value)} />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Scoring…" : "Score draft"}
        </button>
      </form>

      {error && (
        <p className="muted" style={{ marginTop: "1rem", color: "var(--bad)" }}>
          {error} — is <code>MODAL_PREDICT_ENDPOINT</code> set?
        </p>
      )}

      {pred && (
        <>
          <h2>
            Predicted engagement
            <span className="pill">{pred.model_version}</span>
            {pred.used_brain && <span className="pill">brain ✓</span>}
          </h2>
          <EngagementBars pred={pred} />

          <h2>Editor notes</h2>
          {pred.editor_notes.length === 0 ? (
            <p className="muted">No trajectory landmarks flagged.</p>
          ) : (
            <ul className="notes">
              {pred.editor_notes.map((n, i) => (
                <li className="note" key={i}>
                  <span className="k">{n.kind}</span>
                  <span className="m">{n.message}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </main>
  );
}
