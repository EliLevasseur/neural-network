#!/usr/bin/env bash
# Downloads the MNIST handwritten-digit dataset into data/mnist/ and checks
# each file against its published MD5 checksum before unpacking it. Files that
# are already unpacked are skipped. Run from the repository root.
set -euo pipefail

mirror="https://ossci-datasets.s3.amazonaws.com/mnist"
mkdir -p data/mnist
cd data/mnist

md5_of() {
    if command -v md5sum > /dev/null; then
        md5sum "$1" | cut -d ' ' -f 1
    else
        md5 -q "$1"
    fi
}

while read -r expected file; do
    if [ -f "${file%.gz}" ]; then
        continue
    fi
    echo "Downloading $file"
    curl -sSfL -o "$file" "$mirror/$file"
    actual="$(md5_of "$file")"
    if [ "$actual" != "$expected" ]; then
        echo "$file: checksum $actual does not match the published $expected" >&2
        rm -f "$file"
        exit 1
    fi
    gunzip -f "$file"
done <<'CHECKSUMS'
f68b3c2dcbeaaa9fbdd348bbdeb94873 train-images-idx3-ubyte.gz
d53e105ee54ea40749a09fcbcd1e9432 train-labels-idx1-ubyte.gz
9fb629c4189551a2d022fa330f9573f3 t10k-images-idx3-ubyte.gz
ec29112dd5afa0611ce80d1b7f02629c t10k-labels-idx1-ubyte.gz
CHECKSUMS

echo "MNIST is ready in data/mnist/"
