"use client";

import { useEffect, useMemo, useState } from "react";

type UploadSummary = {
  id: string;
  sourceLabel: string | null;
  fileName: string;
  rowCount: number;
  columns: unknown;
  validationIssues: unknown;
  importIssues: unknown;
  importedVideoCount: number;
  processedAt: string | null;
  createdAt: string;
};

const agentPrompt = `You are helping prepare historical video performance data for philo-video-brainlab.

Create one CSV file where each row is one previously posted video. Include our videos and, if available, matched control/competitor videos such as Lectures on Tap, the Met, Crash Course, Veritasium, or Kurzgesagt.

Required columns:
- video_id or external_id
- platform
- url or video_url
- title
- at least one engagement metric: views, likes, comments, shares, saves, reposts, avg_retention, completion_rate, or watch_time_sec

Strongly recommended columns:
- creator or competitor
- posted_at
- duration_sec
- followers_at_post
- caption
- hook_transcript
- full_transcript
- topic
- thumbnail_url

Keep likes, comments, shares, saves, reposts, retention, and watch time as separate columns. Do not collapse them into one score. Save the file as a UTF-8 .csv and tell me exactly where it is stored.`;

export default function TrainingDataPage() {
  const [sourceLabel, setSourceLabel] = useState("Our historical videos");
  const [file, setFile] = useState<File | null>(null);
  const [uploads, setUploads] = useState<UploadSummary[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [copied, setCopied] = useState(false);

  const latestUpload = uploads[0];
  const pendingUploads = uploads.filter((upload) => !upload.processedAt).length;
  const latestColumns = useMemo(() => formatList(latestUpload?.columns), [latestUpload]);
  const latestIssues = useMemo(() => formatList(latestUpload?.validationIssues), [latestUpload]);

  useEffect(() => {
    void loadUploads();
  }, []);

  async function loadUploads() {
    try {
      const res = await fetch("/api/training-uploads", { cache: "no-store" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not load prior uploads");
      setUploads(data.uploads);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load prior uploads");
    }
  }

  async function processUploads() {
    setProcessing(true);
    setMessage(null);
    setError(null);

    try {
      const res = await fetch("/api/training-uploads/process", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not process uploads");
      const result = data.result;
      setMessage(
        `Processed ${result.processedUploads} upload(s) and imported ${result.importedVideos} video row(s).`,
      );
      if (result.issues.length > 0) {
        setError(result.issues.slice(0, 3).join(" "));
      }
      await loadUploads();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not process uploads");
    } finally {
      setProcessing(false);
    }
  }

  async function copyPrompt() {
    await navigator.clipboard.writeText(agentPrompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }

    const form = new FormData();
    form.set("sourceLabel", sourceLabel);
    form.set("file", file);
    setLoading(true);
    setMessage(null);
    setError(null);

    try {
      const res = await fetch("/api/training-uploads", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setUploads((current) => [data.upload, ...current]);
      setMessage(`Saved ${data.upload.rowCount} video rows from ${data.upload.fileName}.`);
      setFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Training data</h1>
      <p className="lede">
        Add old videos and their real engagement metrics here. This is the dataset that turns the
        brain trajectory pipeline from a heuristic scorer into a trained predictor.
      </p>

      <section className="split">
        <div>
          <h2>Upload spreadsheet</h2>
          <form onSubmit={onSubmit}>
            <label>
              Dataset label
              <input value={sourceLabel} onChange={(e) => setSourceLabel(e.target.value)} />
            </label>
            <label>
              CSV file
              <input
                accept=".csv,text/csv"
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Saving..." : "Save training CSV"}
            </button>
          </form>
          {message && <p className="muted success">{message}</p>}
          {error && <p className="muted danger">{error}</p>}
        </div>

        <div>
          <h2>Prompt for your agent</h2>
          <div className="prompt-box">
            <pre>{agentPrompt}</pre>
          </div>
          <button className="secondary" type="button" onClick={copyPrompt}>
            {copied ? "Copied" : "Copy prompt"}
          </button>
        </div>
      </section>

      <h2>Stored uploads</h2>
      <div className="toolbar">
        <p className="muted">
          {pendingUploads === 0 ? "No pending uploads." : `${pendingUploads} upload(s) ready to process.`}
        </p>
        <button className="secondary" type="button" onClick={processUploads} disabled={processing || pendingUploads === 0}>
          {processing ? "Processing..." : "Process pending uploads"}
        </button>
      </div>
      {uploads.length === 0 ? (
        <p className="muted">No training CSVs saved yet.</p>
      ) : (
        <div className="upload-list">
          {uploads.map((upload) => (
            <article className="card" key={upload.id}>
              <h3>{upload.sourceLabel || upload.fileName}</h3>
              <p>{upload.rowCount} video rows saved from {upload.fileName}</p>
              <p>
                {upload.processedAt
                  ? `${upload.importedVideoCount} video row(s) imported`
                  : "Waiting to be processed"}
              </p>
              <p className="muted">{new Date(upload.createdAt).toLocaleString()}</p>
              {upload.processedAt && (
                <p className="muted">Processed {new Date(upload.processedAt).toLocaleString()}</p>
              )}
              {formatList(upload.importIssues) && <p className="muted danger">{formatList(upload.importIssues)}</p>}
            </article>
          ))}
        </div>
      )}

      {latestUpload && (
        <>
          <h2>Latest check</h2>
          <div className="card">
            <p className="muted">Columns found</p>
            <p>{latestColumns || "No columns reported."}</p>
            <p className="muted" style={{ marginTop: ".9rem" }}>Notes</p>
            <p>{latestIssues || "Looks ready for the first training pass."}</p>
          </div>
        </>
      )}
    </main>
  );
}

function formatList(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value.map(String).join(", ");
}
