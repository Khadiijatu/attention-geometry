"""
src/collect.py
==============
YouTube Data API v3 collection pipeline for the attention-geometry project.

Collects video metadata and engagement snapshots for a stratified sample of
800 videos across 10 content categories. Designed for reproducibility:
all parameters are configurable, all outputs are timestamped, and the
collection log records provenance for every video.

Usage
-----
    # Set API key in .env first, then:
    python src/collect.py --n_per_cat 80 --out_dir data/raw

    # Quick test with 5 videos per category:
    python src/collect.py --n_per_cat 5 --out_dir data/raw/test

Author: Khadidiatou Cissé
Date:   April 2026
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "YOUTUBE_API_KEY not found. "
        "Create a .env file with: YOUTUBE_API_KEY=your_key_here"
    )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Category configuration ────────────────────────────────────────────────────
# YouTube category IDs (verified for YouTube Data API v3).
# Each entry: (category_id, human_name, search_query_refinement)
#
# We use both categoryId filtering AND keyword refinement because YouTube's
# category assignment is noisy. The keyword refinement increases precision.

CATEGORIES = [
    {
        "id":        "27",
        "name":      "science_education",
        "label":     "Science & Education",
        "keywords":  "science education explained",
        "duration":  "medium",   # 4–20 min (videoDuration parameter)
    },
    {
        "id":        "23",
        "name":      "comedy",
        "label":     "Comedy",
        "keywords":  "comedy sketch funny",
        "duration":  "short",
    },
    {
        "id":        "26",
        "name":      "wellness",
        "label":     "Meditation & Wellness",
        "keywords":  "meditation mindfulness wellness",
        "duration":  "medium",
    },
    {
        "id":        "25",   # News & Politics — used for finance too
        "name":      "finance",
        "label":     "Finance & Investing",
        "keywords":  "investing personal finance explained",
        "duration":  "medium",
    },
    {
        "id":        "20",
        "name":      "gaming",
        "label":     "Gaming",
        "keywords":  "gaming gameplay walkthrough",
        "duration":  "long",
    },
    {
        "id":        "26",   # Howto & Style
        "name":      "beauty",
        "label":     "Beauty & Fashion",
        "keywords":  "beauty tutorial makeup fashion",
        "duration":  "medium",
    },
    {
        "id":        "26",   # Howto & Style
        "name":      "cooking",
        "label":     "Cooking",
        "keywords":  "recipe cooking tutorial",
        "duration":  "medium",
    },
    {
        "id":        "22",
        "name":      "vlog",
        "label":     "Personal Vlog",
        "keywords":  "vlog daily life personal",
        "duration":  "medium",
    },
    {
        "id":        "25",
        "name":      "politics",
        "label":     "Political Commentary",
        "keywords":  "political commentary analysis opinion",
        "duration":  "medium",
    },
    {
        "id":        "27",
        "name":      "mathematics",
        "label":     "Mathematics & Philosophy",
        "keywords":  "mathematics philosophy lecture proof",
        "duration":  "long",
    },
]

# ── API quota note ────────────────────────────────────────────────────────────
# YouTube Data API v3 costs:
#   search.list         → 100 units per call
#   videos.list         → 1 unit per call
# Daily quota: 10,000 units
#
# For 800 videos (80 per category × 10 categories):
#   search calls:  800/50 = 16 pages × 100 = 1,600 units
#   videos calls:  800 × 1   = 800 units
#   Total:         ~2,400 units   ← well within daily limit
# ─────────────────────────────────────────────────────────────────────────────


def build_service():
    """Initialise the YouTube Data API v3 service."""
    return build("youtube", "v3", developerKey=API_KEY)


def search_videos(service, category: dict, n: int, page_token: str = None):
    """
    Search for videos in a given category.

    Returns a list of video IDs (up to n) and the next page token.
    Uses both category ID and keyword query for precision.
    """
    params = dict(
        part="id",
        type="video",
        videoCategoryId=category["id"],
        q=category["keywords"],
        videoDuration=category["duration"],
        videoEmbeddable="true",
        videoSyndicated="true",
        maxResults=min(n, 50),          # API maximum per page
        relevanceLanguage="en",
        order="relevance",
    )
    if page_token:
        params["pageToken"] = page_token

    response = service.search().list(**params).execute()
    ids = [item["id"]["videoId"] for item in response.get("items", [])]
    next_token = response.get("nextPageToken")
    return ids, next_token


def fetch_video_details(service, video_ids: list[str]) -> list[dict]:
    """
    Fetch full metadata and statistics for a list of video IDs.

    YouTube API allows up to 50 IDs per request (comma-separated).
    Returns one dict per video with all fields normalised.
    """
    if not video_ids:
        return []

    response = service.videos().list(
        part="snippet,statistics,contentDetails,status",
        id=",".join(video_ids),
    ).execute()

    records = []
    for item in response.get("items", []):
        snippet        = item.get("snippet", {})
        stats          = item.get("statistics", {})
        content        = item.get("contentDetails", {})
        status         = item.get("status", {})

        records.append({
            # ── Identity ──────────────────────────────────────────────────
            "video_id":          item["id"],
            "title":             snippet.get("title", ""),
            "description":       snippet.get("description", "")[:1000],  # truncate
            "channel_id":        snippet.get("channelId", ""),
            "channel_title":     snippet.get("channelTitle", ""),
            "category_id":       snippet.get("categoryId", ""),
            "tags":              snippet.get("tags", []),
            "default_language":  snippet.get("defaultLanguage", ""),
            "published_at":      snippet.get("publishedAt", ""),
            "thumbnail_url":     (
                snippet.get("thumbnails", {})
                       .get("maxres", snippet.get("thumbnails", {}).get("high", {}))
                       .get("url", "")
            ),

            # ── Engagement scalars ────────────────────────────────────────
            "view_count":        int(stats.get("viewCount", 0)),
            "like_count":        int(stats.get("likeCount", 0)),
            "comment_count":     int(stats.get("commentCount", 0)),
            "favorite_count":    int(stats.get("favoriteCount", 0)),

            # ── Content metadata ──────────────────────────────────────────
            "duration_iso":      content.get("duration", ""),     # ISO 8601
            "definition":        content.get("definition", ""),   # hd / sd
            "caption":           content.get("caption", ""),      # true / false
            "licensed_content":  content.get("licensedContent", False),

            # ── Status ────────────────────────────────────────────────────
            "privacy_status":    status.get("privacyStatus", ""),
            "made_for_kids":     status.get("madeForKids", False),

            # ── Collection provenance ─────────────────────────────────────
            "collected_at":      datetime.now(timezone.utc).isoformat(),
            "api_version":       "v3",
        })
    return records


def collect_category(service, category: dict, n: int) -> list[dict]:
    """
    Collect n videos for a single category, with pagination and deduplication.
    Handles quota errors with exponential back-off.
    """
    log.info(f"  Category: {category['label']} — target {n} videos")
    collected_ids: set[str] = set()
    all_records:   list[dict] = []
    page_token = None
    attempt = 0

    while len(collected_ids) < n:
        try:
            ids, page_token = search_videos(
                service, category, n=50, page_token=page_token
            )
        except HttpError as e:
            if e.resp.status == 403:
                wait = 2 ** attempt * 10
                log.warning(f"  Quota / rate-limit error. Waiting {wait}s…")
                time.sleep(wait)
                attempt += 1
                if attempt > 5:
                    log.error("  Max retries reached. Stopping category.")
                    break
                continue
            else:
                raise

        # Remove duplicates within this run
        new_ids = [vid for vid in ids if vid not in collected_ids]
        if not new_ids:
            log.info("  No more new results from search.")
            break

        # Fetch details in batches of 50
        for i in range(0, len(new_ids), 50):
            batch = new_ids[i:i+50]
            try:
                records = fetch_video_details(service, batch)
            except HttpError as e:
                log.warning(f"  videos.list error: {e}. Skipping batch.")
                continue

            # Filter: only public, not made-for-kids, has views
            valid = [
                r for r in records
                if r["privacy_status"] == "public"
                and not r["made_for_kids"]
                and r["view_count"] > 0
            ]

            # Tag with our category name (not YouTube's)
            for r in valid:
                r["our_category"] = category["name"]
                r["our_category_label"] = category["label"]

            all_records.extend(valid)
            collected_ids.update(r["video_id"] for r in valid)

            log.info(f"  Collected {len(collected_ids)}/{n} valid videos so far")
            time.sleep(0.5)   # gentle throttle

            if len(collected_ids) >= n:
                break

        if not page_token:
            log.info("  Reached end of search results.")
            break

    # Trim to exactly n
    return all_records[:n]


def parse_duration_seconds(iso: str) -> int:
    """
    Convert ISO 8601 duration string (PT4M13S) to total seconds.

    Examples:
        PT1H2M3S → 3723
        PT45S    → 45
        PT10M    → 600
    """
    import re
    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    m = re.match(pattern, iso)
    if not m:
        return 0
    hours   = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def compute_derived_metrics(record: dict) -> dict:
    """
    Add derived engagement metrics that are used throughout the project.

    These are computed once at collection time and cached in the output.
    All rates are per-view, making them comparable across videos of
    different popularity.
    """
    v = record["view_count"]
    l = record["like_count"]
    c = record["comment_count"]

    record["duration_seconds"] = parse_duration_seconds(record["duration_iso"])
    record["like_rate"]        = l / v if v > 0 else 0.0
    record["comment_rate"]     = c / v if v > 0 else 0.0
    record["engagement_rate"]  = (l + c) / v if v > 0 else 0.0

    # Like-to-comment ratio: high = passive appreciation; low = active discussion
    record["like_comment_ratio"] = l / c if c > 0 else float("inf")

    # Days since publication (approx)
    try:
        pub = datetime.fromisoformat(record["published_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        record["days_since_publication"] = (now - pub).days
    except Exception:
        record["days_since_publication"] = None

    # Views per day (velocity proxy)
    d = record["days_since_publication"]
    record["views_per_day"] = v / d if (d and d > 0) else None

    return record


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    service = build_service()
    log.info("YouTube Data API v3 service initialised.")
    log.info(f"Collecting {args.n_per_cat} videos per category → {out_dir}")

    all_records = []
    collection_log = {
        "started_at":    datetime.now(timezone.utc).isoformat(),
        "n_per_category": args.n_per_cat,
        "categories":    [],
    }

    for cat in CATEGORIES:
        try:
            records = collect_category(service, cat, n=args.n_per_cat)
        except Exception as e:
            log.error(f"Failed category {cat['name']}: {e}")
            records = []

        # Add derived metrics
        records = [compute_derived_metrics(r) for r in records]
        all_records.extend(records)

        # Save per-category file immediately (fault tolerance)
        cat_file = out_dir / f"videos_{cat['name']}.json"
        with open(cat_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        log.info(f"  Saved {len(records)} records → {cat_file}")

        collection_log["categories"].append({
            "name":      cat["name"],
            "collected": len(records),
            "file":      str(cat_file),
        })
        time.sleep(1)   # pause between categories

    # Save combined file
    combined_file = out_dir / "videos_all.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    collection_log["finished_at"]    = datetime.now(timezone.utc).isoformat()
    collection_log["total_collected"] = len(all_records)
    log_file = out_dir / "collection_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(collection_log, f, indent=2)

    log.info("=" * 60)
    log.info(f"Collection complete.")
    log.info(f"  Total videos : {len(all_records)}")
    log.info(f"  Combined file: {combined_file}")
    log.info(f"  Log file     : {log_file}")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect YouTube video metadata for the attention-geometry project."
    )
    parser.add_argument(
        "--n_per_cat",
        type=int,
        default=80,
        help="Number of videos to collect per category (default: 80)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/raw",
        help="Output directory for JSON files (default: data/raw)",
    )
    main(parser.parse_args())
