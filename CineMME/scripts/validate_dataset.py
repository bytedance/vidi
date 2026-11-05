"""Validate dataset structure, source membership, counts, and checksums."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


def normalized_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()) if isinstance(value, str) else ""


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(dataset_dir):
    findings = []
    stats = Counter()
    membership = defaultdict(set)
    question_categories = defaultdict(set)
    datasets = {}
    checksums = {}

    def add(code, dataset, record, field, detail, severity="error"):
        findings.append(dict(code=code, severity=severity, dataset=dataset,
                             record_id=record, field=field, detail=detail))

    for track, filename, id_field in (
        ("grounding", "cinemme_grounding.json", "sample_id"),
        ("mcq", "cinemme_reasoning_mcq.json", "question_id"),
        ("oeq", "cinemme_reasoning_oeq.json", "question_id"),
    ):
        rows = json.loads((dataset_dir / filename).read_text(encoding="utf-8"))
        datasets[track] = rows
        stats[f"{track}_records"] = len(rows)
        seen_ids, seen_videos, seen_questions, seen_segments = set(), set(), {}, set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                add("invalid_record", filename, index, "", "record must be an object")
                continue
            record = row.get(id_field, f"row_{index}")
            required = [id_field, "video_id", "youtube_url"]
            required += ["query", "task", "modality"] if track == "grounding" else ["question", "category", "question_type"]
            required += ["answer"] if track == "mcq" else ["reference_answer", "reasoning_trace", "dense_caption"] if track == "oeq" else []
            for field in required:
                if not normalized_text(row.get(field)):
                    add("empty_or_invalid_text", filename, record, field, row.get(field))
            if isinstance(record, str):
                if record in seen_ids:
                    add("duplicate_id", filename, record, id_field, record)
                seen_ids.add(record)
            video = row.get("video_id")
            if not isinstance(video, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
                add("invalid_video_id", filename, record, "video_id", video)
            else:
                membership[video].add(track)
                if row.get("youtube_url") != f"https://www.youtube.com/watch?v={video}":
                    add("video_url_mismatch", filename, record, "youtube_url", row.get("youtube_url"))
                if track == "grounding" and video in seen_videos:
                    add("duplicate_grounding_video", filename, record, "video_id", video)
                seen_videos.add(video)
            if track != "grounding":
                signature = (str(video), normalized_text(row.get("question")))
                if signature in seen_questions:
                    # Duplicate questions with reordered options are retained in this dataset.
                    add("duplicate_question", filename, record, "question", {"duplicate_of": seen_questions[signature]}, "warning")
                seen_questions[signature] = record
                question_categories[str(row.get("question_type"))].add(str(row.get("category")))
            if track == "mcq":
                options = row.get("options")
                if not isinstance(options, list):
                    add("invalid_options", filename, record, "options", options)
                    continue
                if len(options) != 6:
                    add("option_count", filename, record, "options", {"expected": 6, "actual": len(options)})
                labels, texts = [], defaultdict(list)
                for opt_index, option in enumerate(options):
                    if not isinstance(option, dict):
                        add("invalid_option", filename, record, f"options[{opt_index}]", option)
                        continue
                    label, text = option.get("label"), normalized_text(option.get("text"))
                    labels.append(label)
                    if not text:
                        add("empty_option", filename, record, f"options[{opt_index}].text", option)
                    else:
                        texts[text].append(label)
                if labels != list("ABCDEF"[:len(options)]):
                    add("invalid_option_labels", filename, record, "options", labels)
                if row.get("answer") not in labels:
                    add("invalid_answer", filename, record, "answer", row.get("answer"))
                for text, duplicate_labels in texts.items():
                    if len(duplicate_labels) > 1:
                        add("duplicate_option_text", filename, record, "options", {
                            "labels": duplicate_labels, "includes_gold": row.get("answer") in duplicate_labels, "text": text,
                        })
            if track != "grounding":
                continue
            duration = row.get("duration")
            if not finite_number(duration) or duration <= 0:
                add("invalid_duration", filename, record, "duration", duration)
            segments = row.get("segments")
            if not isinstance(segments, list) or not segments:
                add("empty_or_invalid_segments", filename, record, "segments", segments)
                continue
            stats["grounding_segments"] += len(segments)
            video_box_count = 0
            for seg_index, segment in enumerate(segments):
                field = f"segments[{seg_index}]"
                if not isinstance(segment, dict):
                    add("invalid_segment", filename, record, field, segment)
                    continue
                segment_id = segment.get("segment_id")
                if not isinstance(segment_id, str) or not segment_id or segment_id in seen_segments:
                    add("invalid_or_duplicate_segment_id", filename, record, field, segment_id)
                if isinstance(segment_id, str):
                    seen_segments.add(segment_id)
                if not normalized_text(segment.get("transcript")):
                    add("empty_transcript", filename, record, field, segment_id)
                start, end = segment.get("start"), segment.get("end")
                valid_times = finite_number(start) and finite_number(end)
                if not valid_times or start < 0 or end <= start:
                    add("invalid_segment_interval", filename, record, field, {"segment_id": segment_id, "start": start, "end": end})
                if valid_times and finite_number(duration) and end > duration:
                    excess = end - duration
                    add("segment_after_video", filename, record, field, {"segment_id": segment_id, "end": end, "duration": duration, "excess_seconds": excess}, "warning" if excess <= 0.02 else "error")
                boxes = segment.get("speaker_boxes")
                if not isinstance(boxes, list):
                    add("invalid_boxes", filename, record, field, boxes)
                    continue
                stats["grounding_boxes"] += len(boxes)
                video_box_count += len(boxes)
                stats["segments_without_boxes"] += int(not boxes)
                for box_index, box in enumerate(boxes):
                    box_field = f"{field}.speaker_boxes[{box_index}]"
                    if not isinstance(box, dict):
                        add("invalid_box", filename, record, box_field, box)
                        continue
                    coords, timestamp = box.get("bbox"), box.get("timestamp")
                    if not isinstance(coords, list) or len(coords) != 4 or not all(finite_number(coord) for coord in coords):
                        add("invalid_box", filename, record, box_field, box)
                    else:
                        if not all(0 <= coord <= 1 for coord in coords):
                            add("box_out_of_range", filename, record, box_field, box)
                        if coords[0] >= coords[2] or coords[1] >= coords[3]:
                            add("degenerate_box", filename, record, box_field, box)
                    if not finite_number(timestamp) or timestamp < 0 or (finite_number(duration) and timestamp > duration):
                        add("invalid_box_timestamp", filename, record, box_field, timestamp)
                    elif valid_times and not start <= timestamp <= end:
                        offset = max(start - timestamp, timestamp - end)
                        stats["boxes_outside_segment"] += 1
                        if offset <= 0.100001:
                            stats["boxes_within_100ms_boundary_slack"] += 1
                        else:
                            add("box_outside_segment", filename, record, box_field, {"timestamp": timestamp, "start": start, "end": end, "offset_seconds": offset})
            stats["videos_without_boxes"] += int(video_box_count == 0)
        stats[f"{track}_unique_videos"] = len(seen_videos)

    for question_type, categories in question_categories.items():
        if len(categories) > 1:
            add("inconsistent_category", "reasoning", question_type, "category", sorted(categories), "warning")
    sources = json.loads((dataset_dir / "video_sources.json").read_text(encoding="utf-8"))
    source_ids = set()
    for row in sources:
        video = row["video_id"]
        if video in source_ids:
            add("duplicate_source", "video_sources.json", video, "video_id", video)
        source_ids.add(video)
        if set(row.get("tracks", [])) != membership.get(video, set()) or video not in membership:
            add("source_membership_mismatch", "video_sources.json", video, "tracks", {"actual": row.get("tracks"), "expected": sorted(membership.get(video, []))})
        if row.get("youtube_url") != f"https://www.youtube.com/watch?v={video}":
            add("video_url_mismatch", "video_sources.json", video, "youtube_url", row.get("youtube_url"))
    for video in sorted(membership.keys() - source_ids):
        add("missing_source", "video_sources.json", video, "video_id", video)
    stats["unique_videos"] = len(membership)
    stats["reasoning_unique_videos"] = sum(bool(tracks & {"mcq", "oeq"}) for tracks in membership.values())
    stats["mcq_oeq_shared_videos"] = sum({"mcq", "oeq"} <= tracks for tracks in membership.values())
    stats["grounding_reasoning_shared_videos"] = sum("grounding" in tracks and bool(tracks & {"mcq", "oeq"}) for tracks in membership.values())
    expected_stats = {
        "grounding_samples": stats["grounding_records"], "grounding_segments": stats["grounding_segments"],
        "mcq_questions": stats["mcq_records"], "oeq_questions": stats["oeq_records"], "unique_videos": stats["unique_videos"],
    }
    released_stats = json.loads((dataset_dir / "release_stats.json").read_text(encoding="utf-8"))
    for field, expected in expected_stats.items():
        if released_stats.get(field) != expected:
            add("release_count_mismatch", "release_stats.json", field, field, {"expected": expected, "actual": released_stats.get(field)})
    metadata = json.loads((dataset_dir / "croissant.json").read_text(encoding="utf-8"))
    for entry in metadata["distribution"]:
        filename = entry["name"]
        digest = hashlib.sha256((dataset_dir / filename).read_bytes()).hexdigest()
        checksums[filename] = digest
        if digest != entry.get("sha256"):
            add("checksum_mismatch", "croissant.json", filename, "sha256", {"actual": digest, "declared": entry.get("sha256")})
    for field in ("url", "creator"):
        if "anonymous" in json.dumps(metadata.get(field, "")).lower():
            add("anonymous_metadata", "croissant.json", field, field, metadata[field], "warning")
    return {
        "stats": dict(stats), "sha256": checksums,
        "mcq_answer_distribution": dict(Counter(row.get("answer") for row in datasets["mcq"] if isinstance(row, dict))),
        "counts_by_severity": dict(Counter(item["severity"] for item in findings)),
        "counts_by_code": dict(Counter(item["code"] for item in findings)),
        "findings": findings,
        "scope": "Static annotation checks only; no video availability checks or semantic answer verification. Empty face-box lists are permitted. Boundary offsets <= 100 ms are counted separately; video-end overshoot <= 20 ms is a warning, not an automatic correction.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("findings", "sha256")}, indent=2))
    return int(report["counts_by_severity"].get("error", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
