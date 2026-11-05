#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:?Usage: scripts/run_face_tracking.sh path/to/video.mp4 runs/faces/video_face_tracklets.json}
OUTPUT=${2:?Usage: scripts/run_face_tracking.sh path/to/video.mp4 runs/faces/video_face_tracklets.json}

python -m data_process.face.run_insightface_tracking \
  --video "$VIDEO" \
  --output "$OUTPUT"
