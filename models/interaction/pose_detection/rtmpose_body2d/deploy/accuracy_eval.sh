#!/usr/bin/env bash
# RTMPose-M COCO-WholeBody 三后端正式精度评测驱动 (在 89 宿主机执行).
# 用法: EVAL_RUN_ID=<run_id> bash accuracy_eval.sh [prepare|pytorch|onnx|evaluate|compare|all]
# 全量评测较耗时,必须由用户在前台手动执行并观察实时进度.

set -euo pipefail

readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly CONTAINER="${MTK_G720_CONTAINER:-hhl_g720_8011}"
readonly MODEL_DIR="${PROJECT_ROOT}/models/interaction/pose_detection/rtmpose_body2d"
readonly EVAL_PY="${PROJECT_ROOT}/tools/accuracy/rtmpose_wholebody.py"
readonly METRIC_PY="${MODEL_DIR}/deploy/inference_demo/evaluate_coco_wholebody.py"
readonly MANIFEST_PY="${MODEL_DIR}/deploy/inference_demo/prepare_eval_manifest.py"
readonly COCO_ROOT="${COCO_ROOT:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017}"
readonly IMAGES="${COCO_ROOT}/images"
readonly ANNOTATIONS="${COCO_ROOT}/annotations/coco_wholebody_val_v1.0.json"
readonly DETECTIONS="${COCO_ROOT}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
readonly WEIGHTS_NAME="rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
readonly WEIGHTS="${MODEL_DIR}/models/${WEIGHTS_NAME}"
readonly ONNX_MODEL="${MODEL_DIR}/models/model_fp32.onnx"
readonly CONFIG_RELATIVE="configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"
readonly NPU_RUN_ID="${NPU_RUN_ID:-20260910_mmpose_official_v1}"
readonly NPU_DIR="${MODEL_DIR}/examples/output/board_cpp_accuracy/${NPU_RUN_ID}"
readonly NPU_METRICS="${NPU_DIR}/coco_wholebody_metrics.json"
readonly START="${START:-0}"
readonly TOTAL="${TOTAL:-104125}"
readonly STAGE="${1:-all}"
readonly PYTORCH_DEVICE="${PYTORCH_DEVICE:-cuda:0}"
readonly ONNX_PROVIDER="${ONNX_PROVIDER:-cuda}"
readonly EXPECTED_WEIGHTS_SHA256="3da02694cd6479d3b333ff42ebd0723f96bfa06adac1db1e2e815ed2e9e1b02d"
readonly EXPECTED_ANNOTATIONS_SHA256="f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e"
readonly EXPECTED_DETECTIONS_SHA256="53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59"

if [[ -z "${EVAL_RUN_ID:-}" ]]; then
    echo "必须显式设置 EVAL_RUN_ID,防止长时间结果写入临时运行目录." >&2
    exit 1
fi
if [[ ! "${EVAL_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "EVAL_RUN_ID 必须以字母或数字开头,且只能包含字母、数字、点、下划线和连字符." >&2
    exit 1
fi
if ! [[ "${START}" =~ ^[0-9]+$ ]] || ! [[ "${TOTAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "START 必须为非负整数,TOTAL 必须为正整数." >&2
    exit 1
fi
readonly EVAL_RUN_ID
readonly WORK="${PROJECT_ROOT}/.eval/rtmpose_body2d/runs/${EVAL_RUN_ID}"
readonly MANIFEST="${WORK}/person_detections.tsv"
readonly RUN_CONFIG="${WORK}/run_config.txt"
readonly RUN_HASHES="${WORK}/run_inputs_sha256.txt"
readonly TEMP_ROOT="${TMP_ROOT:-/tmp/hailongcodex/$(date +%Y%m%d)}"


# 检查固定容器处于运行状态,不在脚本中隐式启动容器.
require_running_container() {
    local running
    running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)"
    if [[ "${running}" != "true" ]]; then
        echo "固定容器未运行: ${CONTAINER}" >&2
        echo "请先手动执行: docker start ${CONTAINER}" >&2
        exit 1
    fi
}


# 在固定容器内运行统一 RTMPose 精度工具.
docker_eval() {
    docker exec "${CONTAINER}" python3 "${EVAL_PY}" "$@"
}


# 校验正式模型和数据文件的锁定 SHA-256.
verify_locked_inputs() {
    test -f "${WEIGHTS}"
    test -f "${ONNX_MODEL}"
    test -f "${ANNOTATIONS}"
    test -f "${DETECTIONS}"
    test -f "${NPU_METRICS}"
    echo "${EXPECTED_WEIGHTS_SHA256}  ${WEIGHTS}" | sha256sum --check --status
    echo "${EXPECTED_ANNOTATIONS_SHA256}  ${ANNOTATIONS}" | sha256sum --check --status
    echo "${EXPECTED_DETECTIONS_SHA256}  ${DETECTIONS}" | sha256sum --check --status
}


# 固化本次运行参数和输入哈希,阻止同一 run_id 混用不同输入.
initialize_run() {
    local candidate
    mkdir -p "${WORK}" "${TEMP_ROOT}"
    candidate="$(mktemp "${TEMP_ROOT}/rtmpose_eval_config.XXXXXX")"
    {
        echo "run_id=${EVAL_RUN_ID}"
        echo "start=${START}"
        echo "total=${TOTAL}"
        echo "pytorch_device=${PYTORCH_DEVICE}"
        echo "onnx_provider=${ONNX_PROVIDER}"
        echo "npu_run_id=${NPU_RUN_ID}"
        echo "config_relative=${CONFIG_RELATIVE}"
    } > "${candidate}"
    if [[ -f "${RUN_CONFIG}" ]]; then
        cmp -s "${candidate}" "${RUN_CONFIG}" || {
            echo "运行参数与已有 EVAL_RUN_ID 不一致,请使用新的运行 ID." >&2
            exit 1
        }
        rm -f "${candidate}"
    else
        mv "${candidate}" "${RUN_CONFIG}"
    fi
    if [[ -f "${RUN_HASHES}" ]]; then
        sha256sum --check --status "${RUN_HASHES}" || {
            echo "模型、数据或评测脚本已变化,请使用新的 EVAL_RUN_ID." >&2
            exit 1
        }
    else
        sha256sum "${WEIGHTS}" "${ONNX_MODEL}" "${ANNOTATIONS}" \
            "${DETECTIONS}" "${NPU_METRICS}" "${EVAL_PY}" \
            "${METRIC_PY}" "${MANIFEST_PY}" > "${RUN_HASHES}"
    fi
}


# 生成与板端评测完全相同且顺序固定的人体框清单.
run_prepare() {
    echo "[1/5] 准备 104125 个人体检测框清单."
    if [[ ! -f "${MANIFEST}" ]]; then
        docker exec "${CONTAINER}" python3 "${MANIFEST_PY}" \
            --detections "${DETECTIONS}" --output "${MANIFEST}"
    fi
    local manifest_count
    manifest_count="$(($(wc -l < "${MANIFEST}") - 1))"
    [[ "${manifest_count}" -eq 104125 ]]
    touch "${WORK}/prepare.done"
    echo "[OK] 检测框清单完成: ${manifest_count}/104125."
}


# 执行指定 FP32 后端,预测 JSONL 自带断点续跑能力和实时进度条.
run_backend() {
    local backend="$1"
    test -f "${WORK}/prepare.done"
    echo "[2/5] 执行 ${backend} FP32 推理: START=${START}, TOTAL=${TOTAL}."
    docker_eval infer --backend "${backend}" --model-dir "${MODEL_DIR}" \
        --weights "${WEIGHTS}" --onnx "${ONNX_MODEL}" \
        --config-relative "${CONFIG_RELATIVE}" --manifest "${MANIFEST}" \
        --images "${IMAGES}" \
        --predictions "${WORK}/${backend}_predictions.jsonl" \
        --processed-ids "${WORK}/${backend}_processed_ids.txt" \
        --timing "${WORK}/${backend}_timing.json" --start "${START}" \
        --total "${TOTAL}" --device "${PYTORCH_DEVICE}" \
        --onnx-provider "${ONNX_PROVIDER}"
    [[ "$(wc -l < "${WORK}/${backend}_processed_ids.txt")" -eq "${TOTAL}" ]]
    touch "${WORK}/${backend}.done"
}


# 使用统一后处理分别计算 PyTorch 和 ONNX 的 WholeBody AP/AR.
run_evaluate() {
    local backend
    if [[ "${START}" -ne 0 || "${TOTAL}" -ne 104125 ]]; then
        echo "正式 WholeBody 指标只允许 START=0、TOTAL=104125." >&2
        exit 1
    fi
    for backend in pytorch onnx; do
        test -f "${WORK}/${backend}.done"
        echo "[3/5] 计算 ${backend} COCO-WholeBody 指标."
        docker exec "${CONTAINER}" python3 "${METRIC_PY}" \
            --annotations "${ANNOTATIONS}" \
            --predictions "${WORK}/${backend}_predictions.jsonl" \
            --formatted "${WORK}/${backend}_wholebody_predictions.json" \
            --metrics "${WORK}/${backend}_metrics.json" \
            --summary-log "${WORK}/${backend}_summary.log" 2>&1 \
            | tee "${WORK}/${backend}_evaluate.log"
    done
    touch "${WORK}/evaluate.done"
}


# 汇总三后端同协议精度并固化输出哈希.
run_compare() {
    if [[ "${START}" -ne 0 || "${TOTAL}" -ne 104125 ]]; then
        echo "三后端正式对比只允许 START=0、TOTAL=104125." >&2
        exit 1
    fi
    test -f "${WORK}/evaluate.done"
    echo "[4/5] 汇总 PyTorch、FP32 ONNX 和 MTK NPU INT8 指标."
    docker_eval compare \
        --pytorch-metrics "${WORK}/pytorch_metrics.json" \
        --onnx-metrics "${WORK}/onnx_metrics.json" \
        --npu-metrics "${NPU_METRICS}" \
        --comparison-json "${WORK}/backend_accuracy_comparison.json" \
        --comparison-md "${WORK}/backend_accuracy_comparison.md"
    echo "[5/5] 固化评测输出 SHA-256."
    sha256sum "${WORK}/pytorch_predictions.jsonl" \
        "${WORK}/onnx_predictions.jsonl" \
        "${WORK}/pytorch_metrics.json" "${WORK}/onnx_metrics.json" \
        "${WORK}/backend_accuracy_comparison.json" \
        "${WORK}/backend_accuracy_comparison.md" > "${WORK}/run_outputs_sha256.txt"
    cat "${WORK}/backend_accuracy_comparison.md"
}


require_running_container
verify_locked_inputs
initialize_run
echo "[INFO] 运行 ID: ${EVAL_RUN_ID}"
echo "[INFO] 结果目录: ${WORK}"
case "${STAGE}" in
    prepare) run_prepare ;;
    pytorch) run_backend pytorch ;;
    onnx) run_backend onnx ;;
    evaluate) run_evaluate ;;
    compare) run_compare ;;
    all)
        run_prepare
        run_backend pytorch
        run_backend onnx
        run_evaluate
        run_compare
        ;;
    *)
        echo "未知阶段: ${STAGE}" >&2
        exit 1
        ;;
esac
echo "[OK] RTMPose 三后端评测阶段结束: ${STAGE}, run_id=${EVAL_RUN_ID}."
