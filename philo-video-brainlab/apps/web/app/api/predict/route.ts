import { NextResponse } from "next/server";
import type { EngagementPrediction } from "@/lib/types";

export const runtime = "nodejs";

// Proxies the pre-publication scoring request to the Modal serve endpoint.
// If MODAL_PREDICT_ENDPOINT is unset (local dev before deploy), returns a
// transparent local stub so the UI is fully explorable.
export async function POST(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  const videoId = typeof body.video_id === "string" ? body.video_id.trim() : "";
  if (!videoId) {
    return NextResponse.json({ error: "video_id is required." }, { status: 400 });
  }

  const endpoint = process.env.MODAL_PREDICT_ENDPOINT;

  if (endpoint) {
    const controller = new AbortController();
    const timeoutMs = Number(process.env.MODAL_PREDICT_TIMEOUT_MS ?? 120000);
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(normalizeEndpoint(endpoint), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, video_id: videoId }),
        signal: controller.signal,
      });

      const text = await res.text();
      let data: unknown;
      try {
        data = text ? JSON.parse(text) : {};
      } catch {
        return NextResponse.json(
          { error: "Modal returned a non-JSON response.", status: res.status },
          { status: 502 },
        );
      }

      return NextResponse.json(data, { status: res.status });
    } catch (err) {
      const timedOut = err instanceof DOMException && err.name === "AbortError";
      return NextResponse.json(
        {
          error: timedOut ? "Modal prediction timed out." : "Modal prediction failed.",
        },
        { status: timedOut ? 504 : 502 },
      );
    } finally {
      clearTimeout(timeout);
    }
  }

  // ---- local stub (no Modal endpoint configured) ----
  const seed = hash(videoId);
  const rnd = mulberry32(seed);
  const s = () => Number(rnd().toFixed(2));
  const stub: EngagementPrediction = {
    video_id: videoId,
    model_version: "local-stub-0.1",
    used_brain: false,
    likes: s(),
    comments: s(),
    shares: s(),
    saves: s(),
    retention: s(),
    latent_score: s(),
    editor_notes: [
      {
        tSec: 17,
        kind: "curiosity",
        message: "Curiosity collapses after second 17 (representational change flattens).",
        severity: 2,
      },
    ],
  };
  return NextResponse.json(stub);
}

function normalizeEndpoint(endpoint: string): string {
  return endpoint.endsWith("/") ? endpoint : `${endpoint}/`;
}

function hash(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
