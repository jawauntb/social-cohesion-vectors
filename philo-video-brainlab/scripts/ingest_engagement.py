#!/usr/bin/env python3
"""Ingest a videos manifest into Postgres.

    python scripts/ingest_engagement.py data/manifests/videos.example.json

Every video becomes one row plus its metrics/caption/comments. Keeps all four+
engagement targets separate (see packages/scoring). This uses raw SQL via
psycopg so it stays independent of the Node/Prisma toolchain; run
`npm run db:migrate` first so the schema exists.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import psycopg  # psycopg 3
except ImportError:  # pragma: no cover
    sys.exit("pip install 'psycopg[binary]' first")


def ingest(manifest_path: str) -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("DATABASE_URL not set (see .env.example)")

    rows = json.loads(Path(manifest_path).read_text())
    print(f"Ingesting {len(rows)} videos from {manifest_path}")

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for r in rows:
            comp = cur.execute(
                'SELECT id FROM "Competitor" WHERE name = %s', (r["competitor"],)
            ).fetchone()
            if not comp:
                print(f"  ! unknown competitor {r['competitor']!r} — run db:seed or add it")
                continue
            competitor_id = comp[0]

            cur.execute(
                """
                INSERT INTO "Video" (id, "competitorId", "externalId", platform, url,
                                     title, topic, "durationSec", "followersAtPost")
                VALUES (gen_random_uuid()::text, %s, %s, %s::"Platform", %s, %s, %s, %s, %s)
                ON CONFLICT (platform, "externalId") DO UPDATE SET title = EXCLUDED.title
                RETURNING id
                """,
                (competitor_id, r["externalId"], r["platform"], r.get("url"), r["title"],
                 r.get("topic"), r.get("durationSec"), r.get("followersAtPost")),
            )
            video_id = cur.fetchone()[0]

            m = r.get("metrics", {})
            if m:
                cur.execute(
                    """
                    INSERT INTO "Metric" (id, "videoId", views, likes, comments, shares,
                                          saves, reposts, "avgRetention", "completionRate")
                    VALUES (gen_random_uuid()::text, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (video_id, m.get("views"), m.get("likes"), m.get("comments"),
                     m.get("shares"), m.get("saves"), m.get("reposts"),
                     m.get("avgRetention"), m.get("completionRate")),
                )

            cap = r.get("caption")
            if cap:
                cur.execute(
                    """
                    INSERT INTO "Caption" (id, "videoId", "hookTranscript", "fullTranscript")
                    VALUES (gen_random_uuid()::text, %s, %s, %s)
                    ON CONFLICT ("videoId") DO NOTHING
                    """,
                    (video_id, cap.get("hookTranscript"), cap.get("fullTranscript")),
                )

            for c in r.get("comments", []):
                cur.execute(
                    """
                    INSERT INTO "Comment" (id, "videoId", author, text, "likeCount")
                    VALUES (gen_random_uuid()::text, %s, %s, %s, %s)
                    """,
                    (video_id, c.get("author"), c["text"], c.get("likeCount")),
                )
            print(f"  ✓ {r['title']}")
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ingest(sys.argv[1])
