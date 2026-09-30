#!/bin/bash
set -e

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SRC_DIR/benchmark_onnx_watchdog.sh" /root/hailong.he/ort/benchmark_onnx_watchdog.sh
chmod +x /root/hailong.he/ort/benchmark_onnx_watchdog.sh

cp "$SRC_DIR/onnx-benchmark.service" /etc/systemd/system/onnx-benchmark.service

systemctl daemon-reload
systemctl enable onnx-benchmark.service
systemctl restart onnx-benchmark.service

echo
echo "Installed and started."
echo "Status:"
systemctl --no-pager status onnx-benchmark.service || true
