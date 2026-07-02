import { NextResponse } from "next/server";
import { prisma } from "@brainlab/db";
import { processPendingTrainingUploads } from "@/lib/trainingCsv";

export const runtime = "nodejs";

export async function POST() {
  try {
    const result = await processPendingTrainingUploads(prisma);
    return NextResponse.json({ result });
  } catch {
    return NextResponse.json(
      { error: "Could not process training uploads. Check DATABASE_URL and the Prisma schema." },
      { status: 503 },
    );
  }
}
