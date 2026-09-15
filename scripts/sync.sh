#!/bin/bash
# push: copy code + data to the cluster scratch dir.   pull: copy results back here.
set -e
REMOTE=slurm-login.csail.mit.edu:/data/scratch-oc40/jda/contract
HERE=$(cd "$(dirname "$0")/.." && pwd)
case "$1" in
  push) rsync -az --delete "$HERE/contract" "$HERE/scripts" "$HERE/data" $REMOTE/ ;;
  pull) rsync -az $REMOTE/results/ "$HERE/results/" ;;
  *) echo "usage: sync.sh push|pull"; exit 1 ;;
esac
