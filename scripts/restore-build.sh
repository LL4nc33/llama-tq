#!/bin/bash
# Restore a previously snapshotted build (5 seconds instead of 2-3h rebuild).
#
# Usage:
#   bash scripts/restore-build.sh <sha-or-tag>
#   bash scripts/restore-build.sh --list   # show available snapshots

set -e

REPO=$(git rev-parse --show-toplevel)
BUILD=$REPO/build
SNAPS=$BUILD/snapshots

if [ "$1" = "--list" ] || [ -z "$1" ]; then
    echo "Available snapshots in $SNAPS:"
    if [ -d "$SNAPS" ]; then
        ls -lt "$SNAPS"
    else
        echo "  (none — run snapshot-build.sh after a successful build)"
    fi
    exit 0
fi

SRC=$SNAPS/$1
if [ ! -d "$SRC" ]; then
    echo "ERROR: no snapshot for $1 in $SNAPS" >&2
    echo "Available:" >&2
    ls "$SNAPS" 2>/dev/null >&2 || echo "  (none)" >&2
    exit 1
fi

cp -p "$SRC"/* "$BUILD/bin/"
echo "Restored snapshot $1 from $SRC"
ls -lh "$BUILD/bin/llama-bench" "$BUILD/bin"/libggml*.so* 2>/dev/null
