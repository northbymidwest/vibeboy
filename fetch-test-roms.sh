#!/usr/bin/env bash
set -euo pipefail

REPO="c-sp/game-boy-test-roms"
TAG="v7.0"
ASSET="game-boy-test-roms-${TAG}.zip"
DEST_DIR="game-boy-test-roms"

cd "$(dirname "$0")"

if [ -d "$DEST_DIR" ]; then
    echo "Directory '$DEST_DIR' already exists. Remove it first to re-download."
    exit 1
fi

URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
echo "Downloading ${URL}..."
curl -L -o "/tmp/${ASSET}" "$URL"

echo "Unpacking into ${DEST_DIR}/..."
unzip -q "/tmp/${ASSET}" -d "$DEST_DIR"
rm "/tmp/${ASSET}"

echo "Done. Test ROMs are in ${DEST_DIR}/"
