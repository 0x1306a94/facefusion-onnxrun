#!/bin/bash

set -e

DOWNLOAD_BASE_URL="https://github.com/0x1306a94/facefusion-onnxrun/releases/download/weights_model_v1"
SAVE_DIR=$(realpath $(dirname $0)/../weights)

if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p $SAVE_DIR
fi

function download_verify() {
    FINAL_FILE_URL="$DOWNLOAD_BASE_URL/$1"
    SAVE_FILE_PATH="$SAVE_DIR/$1"
    echo "[*] download $1"
    wget "$FINAL_FILE_URL" -O "$SAVE_FILE_PATH"
    md5sum -c <<<"$2 $SAVE_FILE_PATH"
}

download_verify "2dfan4.onnx" "b6d33e0ab221bc9249d558cf0cbe44b0"
download_verify "arcface_w600k_r50.onnx" "80248d427976241cbd1343889ed132b3"
download_verify "inswapper_128.onnx" "a3a155b90354160350efd66fed6b3d80"
download_verify "yoloface_8n.onnx" "bcd3728be297428848c809ae9fb4b701"
download_verify "gfpgan_1.4.onnx" "2f9d93ad985a8f45eb6dc32268a4576d"