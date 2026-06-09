#!/bin/bash
# Snapshot the current llama-bench + libggml-cuda.so binaries tagged by commit-sha.
# Use after a successful build to enable fast rollback (5s restore vs 2-3h rebuild).
#
# Usage:
#   bash scripts/snapshot-build.sh        # snapshot current HEAD
#   bash scripts/snapshot-build.sh <tag>  # snapshot with custom tag
#
# Restore:
#   bash scripts/restore-build.sh <sha-or-tag>
#
# Snapshots are stored in build/snapshots/<sha>/

set -e

REPO=$(git rev-parse --show-toplevel)
BUILD=$REPO/build

if [ -n "$1" ]; then
    TAG="$1"
else
    TAG=$(git -C "$REPO" rev-parse --short HEAD)
fi

DST=$BUILD/snapshots/$TAG
mkdir -p "$DST"

# Copy bench, server, and the heavy libggml-cuda.so
for f in llama-bench llama-server llama-cli; do
    if [ -f "$BUILD/bin/$f" ]; then
        cp -p "$BUILD/bin/$f" "$DST/"
    fi
done

for f in $BUILD/bin/libggml*.so*; do
    if [ -f "$f" ]; then
        cp -p "$f" "$DST/"
    fi
done

echo "Snapshot $TAG saved to $DST"
ls -lh "$DST" | tail -n +2
