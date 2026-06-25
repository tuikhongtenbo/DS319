#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DIR="${ROOT_DIR}/data/images"

mkdir -p "${IMAGE_DIR}"
cd "${IMAGE_DIR}"

if [ ! -f "test2017.zip" ]; then
  wget http://images.cocodataset.org/zips/test2017.zip
fi

if [ ! -d "COCO2017" ]; then
  unzip -q test2017.zip
  mv test2017 COCO2017
fi

echo "COCO2017 images ready at ${IMAGE_DIR}/COCO2017"
