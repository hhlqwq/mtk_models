#!/usr/bin/env bash

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SOURCE_REVISION="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
readonly WEIGHTS_REVISION="03876f8651c73a60fe4c2c48294e09fcb6838fcf"
readonly WEIGHTS_SHA256="715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
readonly UPSTREAM="${MODEL_ROOT}/original/upstream"
readonly WEIGHTS="${MODEL_ROOT}/original/depth_anything_v2_vits.pth"
readonly TEMP_DIR="/tmp/hailongcodex/$(date +%F)"

mkdir -p "${MODEL_ROOT}/original" "${TEMP_DIR}"
if [[ ! -d "${UPSTREAM}/.git" ]]; then
    echo "[SOURCE] 下载官方源码。"
    git init "${UPSTREAM}"
    git -C "${UPSTREAM}" remote add origin \
        https://github.com/DepthAnything/Depth-Anything-V2.git
    git -C "${UPSTREAM}" fetch --depth 1 origin "${SOURCE_REVISION}"
    git -C "${UPSTREAM}" checkout --detach FETCH_HEAD
fi
actual_revision="$(git -C "${UPSTREAM}" rev-parse HEAD)"
test "${actual_revision}" = "${SOURCE_REVISION}"

if [[ ! -s "${WEIGHTS}" ]]; then
    official_url="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/${WEIGHTS_REVISION}/depth_anything_v2_vits.pth"
    mirror_url="https://hf-mirror.com/depth-anything/Depth-Anything-V2-Small/resolve/${WEIGHTS_REVISION}/depth_anything_v2_vits.pth"
    echo "[WEIGHTS] 下载官方 Small 权重。"
    if ! curl -L --fail --show-error --connect-timeout 10 --max-time 300 \
        --output "${TEMP_DIR}/depth_anything_v2_vits.pth" "${official_url}"; then
        echo "[WEIGHTS] 官方域名不可达，使用固定版本镜像地址。"
        curl -L --fail --show-error --connect-timeout 10 --max-time 300 \
            --output "${TEMP_DIR}/depth_anything_v2_vits.pth" "${mirror_url}"
    fi
    printf '%s  %s\n' "${WEIGHTS_SHA256}" \
        "${TEMP_DIR}/depth_anything_v2_vits.pth" | sha256sum --check -
    mv "${TEMP_DIR}/depth_anything_v2_vits.pth" "${WEIGHTS}"
fi
printf '%s  %s\n' "${WEIGHTS_SHA256}" "${WEIGHTS}" | sha256sum --check -
echo "[OK] 官方源码和权重版本已核验。"
