import { NextResponse } from "next/server";
import type { EngagementPrediction } from "@/lib/types";

// Proxies the pre-publication scoring request to the Modal serve endpoint.
// If MODAL_PREDICT_ENDPOINT is unset (local dev before deploy), returns a
// transparent local stub so the UI is fully explorable.
export async function POST(req: Request) {
  const body = await req.json();
  const endpoint = process.env.MODAL_PREDICT_ENDPOINT;

  if (endpoint) {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  }

  // ---- local stub (no Modal endpoint configured) ----
  const seed = hash(String(body.video_id ?? "draft"));
  const rnd = mulberry32(seed);
  const s = () => Number(rnd().toFixed(2));
  const stub: EngagementPrediction = {
    video_id: body.video_id ?? "draft",
    model_version: "local-stub-0.1",
    used_brain: true,
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
