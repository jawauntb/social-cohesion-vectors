import Link from "next/link";

// Dashboard: states the thesis + the MVP gate. Data views (video list, latent
// map) read from Postgres via @brainlab/db once the catalog is ingested.
export default function Home() {
  return (
    <main>
      <h1>
        philo-video-brainlab <span className="pill">MVP</span>
      </h1>
      <p className="lede">
        Predict social engagement from the estimated <strong>brain-response trajectory</strong> a
        viewer takes while watching a video — using Meta&apos;s TRIBE v2 alongside conventional
        multimodal features.
      </p>

      <blockquote className="thesis">
        Stop recreating the content that worked. Recreate the cognitive trajectory that made it
        work.
      </blockquote>

      <h2>The gate</h2>
      <p className="lede">
        The MVP is a falsifiable question, not a promise: <em>do TRIBE-derived brain trajectories
        improve engagement prediction beyond ordinary video/audio/text features?</em> Every result
        is reported as a with-brain vs. without-brain ablation under grouped cross-validation by
        creator.
      </p>

      <div className="grid">
        <div className="card">
          <h3>1 · Ingest</h3>
          <p>Our catalog + a matched Lectures on Tap control (weak / okay / strong) with metrics,
          captions, and comments.</p>
        </div>
        <div className="card">
          <h3>2 · Features</h3>
          <p>Transcript, visual, audio/prosody, editing rhythm, and a TRIBE v2 brain trajectory per
          video.</p>
        </div>
        <div className="card">
          <h3>3 · Targets, separate</h3>
          <p>likes · comments · shares · saves · retention — kept apart before any latent
          combination.</p>
        </div>
        <div className="card">
          <h3>4 · Ablation gate</h3>
          <p>Baseline vs. baseline+brain per target. Uplift decides go / no-go.</p>
        </div>
        <div className="card">
          <h3>5 · Latent map</h3>
          <p>Plot creators in a shared space — which cognitive regimes reliably drive sharing vs.
          comments.</p>
        </div>
        <div className="card">
          <h3>6 · Editor notes</h3>
          <p>&ldquo;Curiosity collapses after second 17.&rdquo; Trajectory landmarks turned into
          actionable feedback.</p>
        </div>
      </div>

      <h2>Try it</h2>
      <p className="lede">
        Score a draft before you post it: <Link href="/predict">pre-publication scoring →</Link>
      </p>
      <p className="muted" style={{ marginTop: "1.5rem" }}>
        Note: TRIBE v2 and the feature extractors run on Modal GPU and are gated on Hugging Face.
        Until weights are wired, the backend returns a deterministic, clearly-marked stub so the
        full pipeline is runnable end-to-end.
      </p>
    </main>
  );
}
