import type { Platform, PrismaClient } from "@brainlab/db";

export type CsvInspection = {
  columns: string[];
  errors: string[];
  issues: string[];
  rowCount: number;
};

export type CsvPreviewRow = {
  caption: string;
  creator: string;
  metrics: string;
  platform: string;
  title: string;
  url: string;
};

export type TrainingImportResult = {
  importedVideos: number;
  processedUploads: number;
  skippedUploads: number;
  issues: string[];
};

type CsvRecord = Record<string, string>;

const ID_COLUMNS = ["video_id", "external_id", "externalid", "url", "video_url", "local_path"];
const EVIDENCE_COLUMNS = ["evidence_url", "screenshot_url", "screenshot_file", "source_screenshot", "notes"];
const METRIC_COLUMNS = [
  "views",
  "likes",
  "comments",
  "shares",
  "saves",
  "reposts",
  "watch_time",
  "watchtimesec",
  "watch_time_sec",
  "watchtimesec",
  "avg_retention",
  "avgretention",
  "completion_rate",
  "completionrate",
];
const TEMPLATE_COLUMNS = [
  "video_id",
  "platform",
  "url",
  "title",
  "creator",
  "views",
  "likes",
  "comments",
  "shares",
  "saves",
  "reposts",
  "avg_retention",
  "completion_rate",
  "evidence_url",
  "screenshot_url",
  "notes",
];

export const TRAINING_CSV_TEMPLATE = `${TEMPLATE_COLUMNS.join(",")}
example-001,youtube,https://example.com/video,Example title,Our videos,10000,800,120,55,80,12,0.62,0.41,https://drive.google.com/evidence-folder,https://drive.google.com/screenshot,"Replace this row before upload"
`;

export function inspectCsv(csvText: string): CsvInspection {
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
    issues.push("Add title if it is visible. Captions/transcripts can be extracted later from the video URL.");
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
  if (!hasAny(normalized, EVIDENCE_COLUMNS)) {
    issues.push("Add evidence_url, screenshot_url, or notes so engagement numbers can be checked later.");
  }
  if (rowCount < 10) {
    issues.push("This is enough to test the pipeline, but useful training batches should usually have 10+ rows per source.");
  }
  const records = recordsFromCsv(csvText);
  const rowsWithoutMetric = records.filter((record) => !hasAnyMetric(record)).length;
  if (rowsWithoutMetric > 0) {
    issues.push(`${rowsWithoutMetric} row(s) have no filled engagement metric and may be skipped by review.`);
  }

  return { columns, errors, issues, rowCount };
}

export function recordsFromCsv(csvText: string): CsvRecord[] {
  const rows = parseCsv(csvText);
  if (rows.length < 2) return [];

  const headers = rows[0].map(normalizeColumn);
  const records: CsvRecord[] = [];
  rows.slice(1).forEach((row) => {
    if (!row.some((cell) => cell.trim())) return;
    const record: CsvRecord = {};
    headers.forEach((header, index) => {
      if (header) record[header] = (row[index] ?? "").trim();
    });
    records.push(record);
  });
  return records;
}

export function previewRowsFromCsv(csvText: string, maxRows = 5): CsvPreviewRow[] {
  return recordsFromCsv(csvText).slice(0, maxRows).map((record) => ({
    caption: first(record, ["caption", "hook_transcript", "full_transcript"]),
    creator: first(record, ["creator", "competitor", "account", "source"]),
    metrics: metricSummary(record),
    platform: first(record, ["platform"]),
    title: first(record, ["title"]) || first(record, ["video_id", "external_id"]),
    url: first(record, ["url", "video_url", "local_path"]),
  }));
}

export async function processPendingTrainingUploads(prisma: PrismaClient): Promise<TrainingImportResult> {
  const uploads = await prisma.trainingUpload.findMany({
    where: { processedAt: null },
    orderBy: { createdAt: "asc" },
  });
  const result: TrainingImportResult = {
    importedVideos: 0,
    processedUploads: 0,
    skippedUploads: 0,
    issues: [],
  };

  for (const upload of uploads) {
    const records = recordsFromCsv(upload.csvText);
    let importedForUpload = 0;
    const uploadIssues: string[] = [];

    for (const [index, record] of records.entries()) {
      try {
        const imported = await importRecord(prisma, record, upload.sourceLabel ?? upload.fileName);
        if (imported) importedForUpload += 1;
      } catch (err) {
        uploadIssues.push(`row ${index + 2}: ${err instanceof Error ? err.message : "could not import row"}`);
      }
    }

    await prisma.trainingUpload.update({
      where: { id: upload.id },
      data: {
        importIssues: uploadIssues,
        importedVideoCount: importedForUpload,
        processedAt: new Date(),
      },
    });

    result.importedVideos += importedForUpload;
    result.processedUploads += 1;
    result.issues.push(...uploadIssues.map((issue) => `${upload.fileName}: ${issue}`));
  }

  return result;
}

async function importRecord(prisma: PrismaClient, record: CsvRecord, fallbackSource: string): Promise<boolean> {
  const platform = parsePlatform(first(record, ["platform"]));
  const externalId = first(record, ["video_id", "external_id", "externalid"]) || first(record, ["url", "video_url", "local_path"]);
  if (!externalId) throw new Error("missing video_id, external_id, url, video_url, or local_path");

  const competitorName = first(record, ["competitor", "creator", "account", "source"]) || fallbackSource || "Uploaded catalog";
  const title = first(record, ["title"]) || first(record, ["caption", "hook_transcript"]) || externalId;
  const url = first(record, ["url", "video_url"]);

  const competitor = await prisma.competitor.upsert({
    where: { name: competitorName },
    update: {
      platform,
      handle: first(record, ["handle"]) || undefined,
    },
    create: {
      name: competitorName,
      handle: first(record, ["handle"]) || null,
      platform,
      isSelf: isSelfSource(competitorName),
    },
  });

  const video = await prisma.video.upsert({
    where: { platform_externalId: { platform, externalId } },
    update: {
      competitorId: competitor.id,
      url: url || undefined,
      title,
      topic: first(record, ["topic"]) || undefined,
      thumbnailUrl: first(record, ["thumbnail_url", "thumbnail"]) || undefined,
      durationSec: parseFloatValue(first(record, ["duration_sec", "duration"])),
      postedAt: parseDateValue(first(record, ["posted_at", "date"])),
      followersAtPost: parseIntValue(first(record, ["followers_at_post", "followers"])),
    },
    create: {
      competitorId: competitor.id,
      externalId,
      platform,
      url: url || null,
      title,
      topic: first(record, ["topic"]) || null,
      thumbnailUrl: first(record, ["thumbnail_url", "thumbnail"]) || null,
      durationSec: parseFloatValue(first(record, ["duration_sec", "duration"])),
      postedAt: parseDateValue(first(record, ["posted_at", "date"])),
      followersAtPost: parseIntValue(first(record, ["followers_at_post", "followers"])),
    },
  });

  await prisma.metric.create({
    data: {
      videoId: video.id,
      views: parseIntValue(first(record, ["views"])),
      likes: parseIntValue(first(record, ["likes"])),
      comments: parseIntValue(first(record, ["comments"])),
      shares: parseIntValue(first(record, ["shares"])),
      saves: parseIntValue(first(record, ["saves"])),
      reposts: parseIntValue(first(record, ["reposts"])),
      watchTimeSec: parseFloatValue(first(record, ["watch_time_sec", "watch_time", "watchtimesec"])),
      avgRetention: parseRatio(first(record, ["avg_retention", "avgretention", "retention"])),
      completionRate: parseRatio(first(record, ["completion_rate", "completionrate"])),
    },
  });

  const hookTranscript = first(record, ["hook_transcript", "hook"]);
  const fullTranscript = first(record, ["full_transcript", "transcript", "caption"]);
  if (hookTranscript || fullTranscript) {
    await prisma.caption.upsert({
      where: { videoId: video.id },
      update: {
        hookTranscript: hookTranscript || undefined,
        fullTranscript: fullTranscript || undefined,
        language: first(record, ["language"]) || undefined,
      },
      create: {
        videoId: video.id,
        hookTranscript: hookTranscript || null,
        fullTranscript: fullTranscript || null,
        language: first(record, ["language"]) || null,
      },
    });
  }

  return true;
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

function first(record: CsvRecord, keys: string[]): string {
  for (const key of keys) {
    const value = record[normalizeColumn(key)];
    if (value) return value;
  }
  return "";
}

function hasAnyMetric(record: CsvRecord): boolean {
  return METRIC_COLUMNS.some((column) => Boolean(record[column]));
}

function metricSummary(record: CsvRecord): string {
  const pairs = [
    ["views", first(record, ["views"])],
    ["likes", first(record, ["likes"])],
    ["comments", first(record, ["comments"])],
    ["shares", first(record, ["shares"])],
    ["saves", first(record, ["saves"])],
  ].filter(([, value]) => value);
  return pairs.map(([label, value]) => `${label}: ${value}`).join(", ");
}

function parsePlatform(value: string): Platform {
  const normalized = value.trim().toLowerCase();
  if (normalized === "youtube" || normalized === "tiktok" || normalized === "instagram" || normalized === "x") {
    return normalized;
  }
  if (normalized === "twitter") return "x";
  return "other";
}

function parseIntValue(value: string): number | null {
  if (!value) return null;
  const cleaned = value.replace(/[%,$\s]/g, "");
  const parsed = Number.parseInt(cleaned, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseFloatValue(value: string): number | null {
  if (!value) return null;
  const cleaned = value.replace(/[%,$\s]/g, "");
  const parsed = Number.parseFloat(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseRatio(value: string): number | null {
  if (!value) return null;
  const parsed = parseFloatValue(value);
  if (parsed === null) return null;
  return value.includes("%") || parsed > 1 ? parsed / 100 : parsed;
}

function parseDateValue(value: string): Date | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function isSelfSource(value: string): boolean {
  const normalized = value.toLowerCase();
  return normalized.includes("our ") || normalized.includes("philo") || normalized.includes("self");
}
