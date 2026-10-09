#!/usr/bin/env bash
# 两步交付: 89 编译量化并上传,92 开发板执行真实音频全量测试.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/board_paths.conf" ]]; then
    source "${SCRIPT_DIR}/board_paths.conf"
    BOARD_PRECISION="${BOARD_PRECISION:-int8}"
    RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
    RESULT_DIR="${BOARD_RESULTS_DIR}/${RUN_ID}"
    test ! -e "${RESULT_DIR}"
    command -v ffmpeg >/dev/null
    python3 -c 'import numpy'
    mkdir -p "${RESULT_DIR}"
    echo '[开发板 1/3] 解码原始 WAV 并生成固定音频窗口.'
    python3 "${SCRIPT_DIR}/board/evaluate.py" --mode prepare \
        --dataset "${BOARD_DATASET_DIR}" --work "${RESULT_DIR}/work" \
        2>&1 | tee "${RESULT_DIR}/prepare.log"
    echo '[开发板 2/3] 常驻模型预热并执行真实 NPU 全量推理.'
    "${SCRIPT_DIR}/board/yamnet_eval" "${SCRIPT_DIR}/models/model_${BOARD_PRECISION}.dla" \
        "${SCRIPT_DIR}/models/runtime_config.csv" "${RESULT_DIR}/work" \
        2>&1 | tee "${RESULT_DIR}/npu.log"
    echo '[开发板 3/3] 同协议精度对比、性能汇总和真实声音示例.'
    python3 "${SCRIPT_DIR}/board/evaluate.py" --mode summarize \
        --dataset "${BOARD_DATASET_DIR}" --work "${RESULT_DIR}/work" \
        --models-dir "${SCRIPT_DIR}/models" --output "${RESULT_DIR}" --run-id "${RUN_ID}" \
        --precision "${BOARD_PRECISION}"
    sha256sum "${SCRIPT_DIR}/models/"* "${SCRIPT_DIR}/board/yamnet_eval" > "${RESULT_DIR}/SHA256SUMS"
    uname -a > "${RESULT_DIR}/system.txt"
    echo "[OK] 板端汇总: ${RESULT_DIR}/summary.json"
    exit 0
fi

# 编译主机配置: 模型资源离线准备,原始数据固定保存在 NAS.
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${MODEL_ROOT}/models"
DATASET_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/ESC-50"
BUILD_WORK_DIR="/tmp/hailongcodex/$(date +%F)/yamnet"
EXPORT_PYTHON="${YAMNET_EXPORT_PYTHON:-${BUILD_WORK_DIR}/venv/bin/python}"
PRECISION="${YAMNET_PRECISION:-fp16}"
[[ "${PRECISION}" == fp16 || "${PRECISION}" == int8 ]]
CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
NCC_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host"
NEURON_INCLUDE="/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/include"
TARGET_LIBS="/data/users/hailong.he/data/MTKG720/cpp_toolchain/genio720-libs"
BOARD_HOST="root@192.168.0.92"
BOARD_DEPLOY_DIR="/root/hailong.he/open_models/yamnet"
BOARD_DATASET_DIR="/root/hailong.he/datasets/ESC-50"
SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
test -s "${MODELS_DIR}/yamnet.h5"
echo '13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744 '"${MODELS_DIR}/yamnet.h5" | sha256sum -c -
test -s "${DATASET_DIR}/meta/esc50.csv"
mkdir -p "${BUILD_WORK_DIR}"
docker exec "${CONTAINER}" test -x "${EXPORT_PYTHON}"
echo '[编译主机 1/4] 官方前向比较、固定 ONNX 导出和三后端参考准备.'
docker exec -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 "${CONTAINER}" \
    "${EXPORT_PYTHON}" "${SCRIPT_DIR}/host/export_evaluate.py" --mode export \
    --models-dir "${MODELS_DIR}" --dataset "${DATASET_DIR}"
docker exec -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 "${CONTAINER}" \
    "${EXPORT_PYTHON}" "${SCRIPT_DIR}/host/export_evaluate.py" --mode tensorflow \
    --models-dir "${MODELS_DIR}" --dataset "${DATASET_DIR}" --output "${BUILD_WORK_DIR}/tensorflow"
docker exec -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 "${CONTAINER}" \
    python "${SCRIPT_DIR}/host/export_evaluate.py" --mode onnx \
    --models-dir "${MODELS_DIR}" --dataset "${DATASET_DIR}" \
    --output "${BUILD_WORK_DIR}/onnx" --reference "${BUILD_WORK_DIR}/tensorflow"

echo "[编译主机 2/4] ${PRECISION} 转换和 MDLA 5.3 DLA 编译."
docker exec -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 "${CONTAINER}" \
    python "${SCRIPT_DIR}/host/convert_int8.py" --models-dir "${MODELS_DIR}" --dataset "${DATASET_DIR}" \
    --precision "${PRECISION}"
NCC_FLAGS=(--arch=mdla5.3 --suppress-output --disallow-bridge)
if [[ "${PRECISION}" == fp16 ]]; then
    NCC_FLAGS+=(--relax-fp32 --suppress-input)
fi
docker exec -e LD_LIBRARY_PATH="${NCC_ROOT}/lib" "${CONTAINER}" \
    "${NCC_ROOT}/bin/ncc-tflite" "${NCC_FLAGS[@]}" \
    "${MODELS_DIR}/model_${PRECISION}.tflite" -o "${MODELS_DIR}/model_${PRECISION}.dla"

echo '[编译主机 3/4] 交叉编译 C++ Runtime 程序并准备参考与示例.'
aarch64-linux-gnu-g++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"${NEURON_INCLUDE}" "${SCRIPT_DIR}/board/yamnet_eval.cpp" \
    "${TARGET_LIBS}/libneuronusdk_runtime.mtk.so.8" -Wl,--allow-shlib-undefined -pthread -ldl \
    -o "${BUILD_WORK_DIR}/yamnet_eval"
docker exec -i -e MODEL_ROOT="${MODEL_ROOT}" -e DATASET_DIR="${DATASET_DIR}" \
    -e BUILD_WORK_DIR="${BUILD_WORK_DIR}" "${CONTAINER}" python - <<'REFERENCE'
import hashlib,json,os,sys,shutil
from pathlib import Path
root=Path(os.environ['MODEL_ROOT']); dataset=Path(os.environ['DATASET_DIR']); work=Path(os.environ['BUILD_WORK_DIR'])
sys.path.insert(0,str(root/'deploy/board'))
from audio_utils import load_records,mapping
labels=root/'models/upstream/yamnet_class_map.csv'
reference={backend:json.loads((work/backend/'summary.json').read_text()) for backend in ['tensorflow','onnx']}
reference['annotations_sha256']=hashlib.sha256((dataset/'meta/esc50.csv').read_bytes()).hexdigest()
reference['mapping']=mapping(labels)
(root/'models/fp32_reference.json').write_text(json.dumps(reference,indent=2))
shutil.copy2(labels,root/'models/yamnet_class_map.csv')
inputs=root/'examples/input'; inputs.mkdir(parents=True,exist_ok=True)
records=load_records(dataset); samples=[]
reference['audio_sha256']={row['filename']:hashlib.sha256((dataset/'audio'/row['filename']).read_bytes()).hexdigest() for row in records}
(root/'models/fp32_reference.json').write_text(json.dumps(reference,indent=2))
for number,target in enumerate([0,39,42],1):
    row=next(row for row in records if int(row['target'])==target)
    shutil.copy2(dataset/'audio'/row['filename'],inputs/f'sample_{number}.wav')
    samples.append(row)
(inputs/'samples.json').write_text(json.dumps(samples,indent=2))
REFERENCE

echo '[编译主机 4/4] 同步板端原始音频、模型、程序及运行入口.'
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DEPLOY_DIR}/models' '${BOARD_DEPLOY_DIR}/board' '${BOARD_DEPLOY_DIR}/examples/input' '${BOARD_DATASET_DIR}'"
rsync -a --info=progress2 "${DATASET_DIR}/audio" "${DATASET_DIR}/meta" \
    "${DATASET_DIR}/LICENSE" "${DATASET_DIR}/source_manifest.json" "${BOARD_HOST}:${BOARD_DATASET_DIR}/"
scp "${SSH_OPTIONS[@]}" "${MODELS_DIR}/model_${PRECISION}.dla" "${MODELS_DIR}/runtime_config.csv" \
    "${MODELS_DIR}/yamnet_class_map.csv" "${MODELS_DIR}/fp32_reference.json" \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/models/"
scp "${SSH_OPTIONS[@]}" "${BUILD_WORK_DIR}/yamnet_eval" "${SCRIPT_DIR}/board/"*.py \
    "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/board/"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run.sh" "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/examples/input/"* "${BOARD_HOST}:${BOARD_DEPLOY_DIR}/examples/input/"
printf 'BOARD_DATASET_DIR=%q\nBOARD_RESULTS_DIR=%q\nBOARD_PRECISION=%q\n' \
    "${BOARD_DATASET_DIR}" "${BOARD_DEPLOY_DIR}/results" "${PRECISION}" |
    ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "cat > '${BOARD_DEPLOY_DIR}/board_paths.conf'"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "chmod 755 '${BOARD_DEPLOY_DIR}/board/yamnet_eval'"
echo "[NEXT] 开发板执行: bash '${BOARD_DEPLOY_DIR}/run.sh'"
