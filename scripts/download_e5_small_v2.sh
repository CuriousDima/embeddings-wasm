#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-model}"
mkdir -p "$TARGET_DIR"

BASE_URL="https://huggingface.co/intfloat/e5-small-v2/resolve/main"

for file in config.json tokenizer.json model.safetensors; do
  echo "Downloading $file ..."
  curl -L "$BASE_URL/$file" -o "$TARGET_DIR/$file"
done

echo "Model files saved into: $TARGET_DIR"
