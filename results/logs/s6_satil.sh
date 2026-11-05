#!/bin/bash
# Satil.zip was dropped at 39% and the SQUID loop moved on; resume until the md5 matches
cd ~/Downloads/MLOE---Group-4/data/external/squid
read -r name md5 url < <(grep '^Satil.zip' files.tsv)
until [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ]; do
  curl -sS -L -C - --retry 8 --retry-delay 20 -o "$name" "$url"; sleep 30
done
echo SATIL_DONE
