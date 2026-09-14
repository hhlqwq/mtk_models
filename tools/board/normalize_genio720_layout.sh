#!/usr/bin/env bash
# 在 Genio 720 板端整理模型目录,不删除任何文件.

set -euo pipefail

readonly BOARD_ROOT="${1:-/root/hailong.he}"
readonly ARCHIVE_DATE="${ARCHIVE_DATE:-$(date +%Y%m%d)}"
readonly ARCHIVE_ROOT="${BOARD_ROOT}/archive/${ARCHIVE_DATE}/layout_normalization"
readonly LEGACY_ROOT="${ARCHIVE_ROOT}/legacy"
readonly VIT_ROOT="${BOARD_ROOT}/vit_base_patch16_224"
readonly YOLO_ROOT="${BOARD_ROOT}/yolov5s"
readonly RTM_ROOT="${BOARD_ROOT}/rtmpose_body2d"


# 输出带阶段的迁移日志.
log() {
    echo "[LAYOUT] $*"
}


# 移动一个已确认存在的文件或目录,目标不能预先存在.
move_if_present() {
    local source="$1"
    local target="$2"
    if [[ ! -e "${source}" ]]; then
        return
    fi
    if [[ -e "${target}" ]]; then
        echo "[ERROR] 迁移目标已存在: ${target}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${target}")"
    log "MOVE ${source} -> ${target}"
    mv -- "${source}" "${target}"
}


# 将来源目录的一层子项逐个移动到目标目录,防止覆盖同名运行 ID.
move_children() {
    local source_dir="$1"
    local target_dir="$2"
    local item=""
    if [[ ! -d "${source_dir}" ]]; then
        return
    fi
    mkdir -p "${target_dir}"
    for item in "${source_dir}"/*; do
        [[ -e "${item}" ]] || continue
        move_if_present "${item}" "${target_dir}/$(basename "${item}")"
    done
    rmdir "${source_dir}"
}


# 创建单模型固定三层目录.
prepare_model_root() {
    local model_root="$1"
    mkdir -p "${model_root}/model" "${model_root}/demo/smoke" \
        "${model_root}/eval"
}


# 检查整理后的模型根目录只含 model、demo、eval.
verify_model_root() {
    local model_root="$1"
    local item=""
    for item in "${model_root}"/*; do
        case "$(basename "${item}")" in
            model|demo|eval) ;;
            *)
                echo "[ERROR] 模型根目录存在未归类项目: ${item}" >&2
                exit 1
                ;;
        esac
    done
}


# 检查板端根目录只保留共享目录和三个标准模型目录.
verify_board_root() {
    local item=""
    shopt -s dotglob nullglob
    for item in "${BOARD_ROOT}"/*; do
        case "$(basename "${item}")" in
            archive|datasets|wheels|vit_base_patch16_224|yolov5s|rtmpose_body2d) ;;
            *)
                echo "[ERROR] 板端根目录存在未归类项目: ${item}" >&2
                exit 1
                ;;
        esac
    done
    shopt -u dotglob nullglob
}


test -d "${BOARD_ROOT}"
if pgrep -x neuronrt >/dev/null 2>&1; then
    echo "[ERROR] 检测到运行中的 neuronrt,请等待任务结束后再整理." >&2
    exit 1
fi
if [[ -e "${ARCHIVE_ROOT}" ]]; then
    echo "[ERROR] 当日归档目录已存在: ${ARCHIVE_ROOT}" >&2
    exit 1
fi

log "创建标准模型目录."
prepare_model_root "${VIT_ROOT}"
prepare_model_root "${YOLO_ROOT}"
prepare_model_root "${RTM_ROOT}"
mkdir -p "${LEGACY_ROOT}"

log "迁移 ViT 正式评测和 Demo."
move_children "${BOARD_ROOT}/vit_eval/runs" "${VIT_ROOT}/eval"
move_if_present "${BOARD_ROOT}/vit_eval" "${LEGACY_ROOT}/vit_eval_residual"
move_if_present "${VIT_ROOT}/model_int8.dla" "${VIT_ROOT}/model/model_int8.dla"
move_if_present "${VIT_ROOT}/input_int8.bin" "${VIT_ROOT}/demo/smoke/input_int8.bin"
move_if_present "${VIT_ROOT}/run_board.sh" "${VIT_ROOT}/demo/smoke/run_board.sh"
move_if_present "${VIT_ROOT}/output" "${VIT_ROOT}/demo/smoke/output"
move_if_present "${BOARD_ROOT}/vit_public_examples" "${VIT_ROOT}/demo/public"

log "迁移 YOLOv5s 正式评测和 Demo."
move_children "${BOARD_ROOT}/yolov5s_cpp/runs" "${YOLO_ROOT}/eval"
move_if_present "${YOLO_ROOT}/model_int8.dla" "${YOLO_ROOT}/model/model_int8.dla"
move_if_present "${YOLO_ROOT}/model_int8_mdla53.dla" \
    "${YOLO_ROOT}/model/model_int8_mdla53.dla"
move_if_present "${YOLO_ROOT}/input_int8.bin" "${YOLO_ROOT}/demo/smoke/input_int8.bin"
move_if_present "${YOLO_ROOT}/run_board.sh" "${YOLO_ROOT}/demo/smoke/run_board.sh"
move_if_present "${YOLO_ROOT}/output" "${YOLO_ROOT}/demo/smoke/output"
move_if_present "${BOARD_ROOT}/yolov5s_public_examples" "${YOLO_ROOT}/demo/public"
move_if_present "${BOARD_ROOT}/yolov5s_cpp/yolov5s_board_eval" \
    "${YOLO_ROOT}/model/yolov5s_board_eval"
move_if_present "${BOARD_ROOT}/yolov5s_cpp/evaluate_coco.py" \
    "${YOLO_ROOT}/model/evaluate_coco.py"
move_if_present "${BOARD_ROOT}/yolov5s_cpp" "${LEGACY_ROOT}/yolov5s_cpp_residual"
move_if_present "${BOARD_ROOT}/yolov5s_examples" "${LEGACY_ROOT}/yolov5s_examples"
move_if_present "${BOARD_ROOT}/yolov5s_old" "${LEGACY_ROOT}/yolov5s_old"
move_if_present "${BOARD_ROOT}/yolov5s_probe" "${LEGACY_ROOT}/yolov5s_probe"

log "迁移 RTMPose 正式评测和 Demo."
move_children "${BOARD_ROOT}/rtmpose_body2d_cpp/runs" "${RTM_ROOT}/eval"
move_if_present "${RTM_ROOT}/model_int8.dla" "${RTM_ROOT}/model/model_int8.dla"
move_if_present "${RTM_ROOT}/inputs" "${RTM_ROOT}/demo/smoke/inputs"
move_if_present "${RTM_ROOT}/output" "${RTM_ROOT}/demo/smoke/output"
move_if_present "${RTM_ROOT}/run_board.sh" "${RTM_ROOT}/demo/smoke/run_board.sh"
move_if_present "${BOARD_ROOT}/rtmpose_public_examples" "${RTM_ROOT}/demo/public"
move_if_present "${BOARD_ROOT}/rtmpose_body2d_cpp/rtmpose_board_eval" \
    "${RTM_ROOT}/model/rtmpose_board_eval"
move_if_present "${BOARD_ROOT}/rtmpose_body2d_cpp/person_detections.tsv" \
    "${RTM_ROOT}/model/person_detections.tsv"
move_if_present "${BOARD_ROOT}/rtmpose_body2d_cpp" \
    "${LEGACY_ROOT}/rtmpose_body2d_cpp_residual"

log "归档根目录的旧模型、探针和裸模型文件."
for item in \
    "${BOARD_ROOT}/.mount.sh.un~" \
    "${BOARD_ROOT}/model_int8.dla" \
    "${BOARD_ROOT}/np8_srresnet_probe.dla" \
    "${BOARD_ROOT}/probe_int8raw" \
    "${BOARD_ROOT}/yolov5n" \
    "${BOARD_ROOT}/yolov5n_cal_loss" \
    "${BOARD_ROOT}/yolov8s" \
    "${BOARD_ROOT}/yolov8s_cal_loss"; do
    move_if_present "${item}" "${LEGACY_ROOT}/$(basename "${item}")"
done

verify_model_root "${VIT_ROOT}"
verify_model_root "${YOLO_ROOT}"
verify_model_root "${RTM_ROOT}"
verify_board_root
log "整理完成.共享目录保留: datasets、wheels、archive."
