import { TRAINING_CSV_TEMPLATE } from "@/lib/trainingCsv";

export const runtime = "nodejs";

export async function GET() {
  return new Response(TRAINING_CSV_TEMPLATE, {
    headers: {
      "Content-Disposition": "attachment; filename=\"brainlab-training-template.csv\"",
      "Content-Type": "text/csv; charset=utf-8",
    },
  });
}
