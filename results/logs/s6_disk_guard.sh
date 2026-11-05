#!/bin/bash
# Pause the dataset downloads (SIGSTOP, resumable) when free space drops under 8 GB, so a full disk cannot kill training
while sleep 120; do
  free=$(df --output=avail -BG ~/Downloads/MLOE---Group-4 | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 8 ]; then pgrep -x curl | xargs -r kill -STOP; echo "$(date +%T) paused downloads, ${free} GB free"; fi
done
