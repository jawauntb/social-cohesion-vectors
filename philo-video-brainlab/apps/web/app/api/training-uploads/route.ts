import { NextResponse } from "next/server";
import { prisma } from "@brainlab/db";

export const runtime = "nodejs";

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
const ID_COLUMNS = ["video_id", "external_id", "externalid", "url", "video_url", "local_path"];
const METRIC_COLUMNS = [
  "views",
  "likes",
  "comments",
  "shares",
  "saves",
  "reposts",
  "watch_time",
  "watchtimesec",
  "avg_retention",
  "avgretention",
  "completion_rate",
  "completionrate",
];

export async function GET() {
  try {
    const uploads = await prisma.trainingUpload.findMany({
      orderBy: { createdAt: "desc" },
      take: 8,
      select: {
        id: true,
        sourceLabel: true,
        fileName: true,
        rowCount: true,
        columns: true,
        validationIssues: true,
        createdAt: true,
      },
    });
    return NextResponse.json({ uploads });
  } catch {
    return NextResponse.json(
      { error: "Training uploads need DATABASE_URL and the Prisma schema deployed." },
      { status: 503 },
    );
  }
}

export async function POST(req: Request) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return NextResponse.json({ error: "Upload must be sent as form data." }, { status: 400 });
  }

  const file = form.get("file");
  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Choose a CSV file to upload." }, { status: 400 });
  }
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return NextResponse.json({ error: "The training sheet must be a .csv file." }, { status: 400 });
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json({ error: "CSV is too large. Keep the first import under 4 MB." }, { status: 400 });
  }

  const csvText = await file.text();
  const parsed = inspectCsv(csvText);
  if (parsed.errors.length > 0) {
    return NextResponse.json({ error: parsed.errors[0], issues: parsed.errors }, { status: 400 });
  }

  const sourceLabel = stringField(form.get("sourceLabel"));

  try {
    const upload = await prisma.trainingUpload.create({
      data: {
        sourceLabel,
        fileName: file.name,
        rowCount: parsed.rowCount,
        columns: parsed.columns,
        validationIssues: parsed.issues,
        csvText,
      },
      select: {
        id: true,
        sourceLabel: true,
        fileName: true,
        rowCount: true,
        columns: true,
        validationIssues: true,
        createdAt: true,
      },
    });

    return NextResponse.json({ upload }, { status: 201 });
  } catch {
    return NextResponse.json(
      { error: "Could not persist the CSV. Check DATABASE_URL and run the Prisma schema update." },
      { status: 503 },
    );
  }
}

function inspectCsv(csvText: string): {
  columns: string[];
  errors: string[];
  issues: string[];
  rowCount: number;
} {
  const rows = parseCsv(csvText);
  if (rows.length < 2) {
    return { columns: [], errors: ["CSV needs a header row and at least one video row."], issues: [], rowCount: 0 };
  }

  const columns = rows[0].map((cell) => cell.trim()).filter(Boolean);
  const normalized = columns.map(normalizeColumn);
  const rowCount = rows.slice(1).filter((row) => row.some((cell) => cell.trim())).length;
  const errors: string[] = [];
  const issues: string[] = [];

  if (columns.length === 0) errors.push("CSV header row is empty.");
  if (rowCount === 0) errors.push("CSV has no video rows.");
  if (!hasAny(normalized, ID_COLUMNS)) {
    errors.push("CSV needs one identifier column: video_id, external_id, url, video_url, or local_path.");
  }
  if (!hasAny(normalized, ["title", "caption", "hook_transcript", "full_transcript"])) {
    issues.push("Add title, caption, hook_transcript, or full_transcript so the model has text context.");
  }
  if (!hasAny(normalized, ["platform"])) {
    issues.push("Add platform so YouTube/TikTok/Instagram rows can be grouped correctly.");
  }
  if (!hasAny(normalized, METRIC_COLUMNS)) {
    errors.push("CSV needs at least one engagement metric: views, likes, comments, shares, saves, reposts, retention, or watch time.");
  }
  if (!hasAny(normalized, ["competitor", "creator", "account", "source"])) {
    issues.push("Add competitor, creator, account, or source to separate your videos from controls.");
  }

  return { columns, errors, issues, rowCount };
}

function parseCsv(input: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    const next = input[i + 1];

    if (char === "\"") {
      if (inQuotes && next === "\"") {
        cell += "\"";
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(cell);
      cell = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      continue;
    }

    cell += char;
  }

  row.push(cell);
  rows.push(row);
  return rows.filter((csvRow) => csvRow.some((value) => value.trim()));
}

function normalizeColumn(column: string): string {
  return column.trim().toLowerCase().replace(/[\s-]+/g, "_").replace(/[^a-z0-9_]/g, "");
}

function hasAny(columns: string[], candidates: string[]): boolean {
  return candidates.some((candidate) => columns.includes(candidate));
}

function stringField(value: FormDataEntryValue | null): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}
