#!/usr/bin/env bash
# Checks a release archive before it is published:
#   - the main tools are inside (server, cli, bench, quantize, perplexity, finetune)
#   - every binary finds the bundled libraries next to it ($ORIGIN), from any directory
#   - every other library it needs is a common system library or, for CUDA, the CUDA runtime the user installs
#   - CPU and Vulkan archives start (--version) from another directory with an empty environment and report the
#     release version, if one is given
#
# usage: scripts/check-release-archive.sh <archive.tar.gz> <cpu|vulkan|cuda> [release version]
set -euo pipefail

archive=$1
kind=$2
version=${3:-}

dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
tar -xzf "$archive" -C "$dir"
bin=$(find "$dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)

fail=0
err() { echo "ERROR: $*"; fail=1; }

for tool in llama-server llama-cli llama-bench llama-quantize llama-perplexity llama-finetune; do
    [ -x "$bin/$tool" ] || err "missing $tool"
done

# libraries the user's system provides
allowed='^(libc|libm|libdl|librt|libpthread|libstdc\+\+|libgcc_s|libgomp|libssl|libcrypto|ld-linux-x86-64|ld-linux-aarch64)\.so'
case $kind in
    vulkan) allowed="${allowed}|^libvulkan\.so" ;;
    cuda)   allowed="${allowed}|^(libcuda|libcudart|libcublas|libcublasLt)\.so" ;;
esac

for f in "$bin"/*; do
    [ -L "$f" ] && continue
    [ -f "$f" ] && head -c 4 "$f" | grep -q $'\x7fELF' || continue
    name=$(basename "$f")
    runpath=$(readelf -d "$f" | sed -n 's/.*(RUNPATH).*\[\(.*\)\]/\1/p; s/.*(RPATH).*\[\(.*\)\]/\1/p')
    if [ -n "$runpath" ] && [ "$runpath" != '$ORIGIN' ]; then
        err "$name: RUNPATH is '$runpath', expected \$ORIGIN"
    fi
    for lib in $(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
        [ -e "$bin/$lib" ] && continue
        echo "$lib" | grep -Eq "$allowed" || err "$name needs $lib, which is neither bundled nor a common system library"
    done
done

if [ "$kind" != cuda ]; then
    # the CUDA archive needs a GPU driver to start, the others must run anywhere
    out=$(cd / && env -i "$bin/llama-server" --version 2>&1) || err "llama-server --version failed"
    echo "$out"
    if [ -n "$version" ] && ! echo "$out" | grep -Fxq -- "llama-tq $version"; then
        err "llama-server --version does not report llama-tq $version"
    fi
fi

[ $fail -eq 0 ] && echo "OK: $archive"
exit $fail
