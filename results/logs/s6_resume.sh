#!/bin/bash
# Resume a SQUID zip the main loop dropped and the SQUID loop moved on; resume until the md5 matches
cd ~/Downloads/MLOE---Group-4/data/external/squid
read -r name md5 url < <(grep "^$1" files.tsv)
until [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ]; do
  curl -sS -L -C - --retry 8 --retry-delay 20 -o "$name" "$url"; sleep 30
done
echo "RESUME_DONE $1"
