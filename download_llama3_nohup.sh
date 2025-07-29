#!/bin/bash
set -e
DIR=./llama3
mkdir -p "$DIR"

# 要下的四个分片
FILES=(
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json
)

for f in "${FILES[@]}"; do
  while true; do
    echo "==== 下载 $f ===="
    huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
      --local-dir "$DIR" --local-dir-use-symlinks False \
      --include "$f" && break
    echo "下载 $f 失败，30 秒后重试..."
    sleep 30
  done
done

echo "全部文件下载完成："
ls -lh "$DIR"/model-0000*-of-00004.safetensors "$DIR"/model.safetensors.index.json
