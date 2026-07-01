// Shared client/server types for engagement predictions and editor notes.
// Mirrors services/modal/modal_app/schemas.py (EngagementPrediction).

export const TARGETS = ["likes", "comments", "shares", "saves", "retention"] as const;
export type Target = (typeof TARGETS)[number];

export interface EditorNote {
  tSec: number;
  kind: string;
  message: string;
  severity: number;
}

export interface EngagementPrediction {
  video_id: string;
  model_version: string;
  used_brain: boolean;
  likes?: number;
  comments?: number;
  shares?: number;
  saves?: number;
  retention?: number;
  latent_score?: number;
  editor_notes: EditorNote[];
}
