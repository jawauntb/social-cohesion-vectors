import { NextResponse } from "next/server";
import { prisma } from "@brainlab/db";
import { inspectCsv, previewRowsFromCsv } from "@/lib/trainingCsv";

export const runtime = "nodejs";

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

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
        importIssues: true,
        importedVideoCount: true,
        processedAt: true,
        createdAt: true,
        csvText: true,
      },
    });
    const sourceSummary = await prisma.competitor.findMany({
      orderBy: { name: "asc" },
      select: {
        id: true,
        isSelf: true,
        name: true,
        platform: true,
        _count: { select: { videos: true } },
      },
    });
    return NextResponse.json({
      sourceSummary: sourceSummary.map((source) => ({
        id: source.id,
        isSelf: source.isSelf,
        name: source.name,
        platform: source.platform,
        videoCount: source._count.videos,
      })),
      uploads: uploads.map(({ csvText, ...upload }) => ({
        ...upload,
        previewRows: previewRowsFromCsv(csvText),
      })),
    });
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
        importIssues: true,
        importedVideoCount: true,
        processedAt: true,
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

function stringField(value: FormDataEntryValue | null): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}
