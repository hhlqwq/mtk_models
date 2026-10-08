"""反量化 RTMPose SimCC 输出并生成关键点 JSON 与可视化."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BODY_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
    (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)


def load_output(path: Path, detail: dict) -> np.ndarray:
    """按元数据读取并反量化一份 MDLA 原生输出."""
    dtype = np.dtype(detail["dtype"])
    raw = np.fromfile(path, dtype=dtype)
    shape = tuple(detail["shape"])
    expected_size = int(np.prod(shape))
    if raw.size != expected_size:
        raise ValueError(
            f"输出大小错误: {path}, 期望 {expected_size}, 实际 {raw.size}")
    return ((raw.reshape(shape).astype(np.float32) - detail["zero_point"])
            * detail["scale"])


def decode_simcc(pred_x: np.ndarray,
                 pred_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按 split ratio 2.0 解码 133 点 SimCC 坐标和分数."""
    x_locations = np.argmax(pred_x, axis=-1).astype(np.float32)
    y_locations = np.argmax(pred_y, axis=-1).astype(np.float32)
    x_scores = np.max(pred_x, axis=-1)
    y_scores = np.max(pred_y, axis=-1)
    keypoints = np.stack((x_locations, y_locations), axis=-1) / 2.0
    return keypoints[0], np.minimum(x_scores, y_scores)[0]


def map_to_source(keypoints: np.ndarray, sample: dict) -> np.ndarray:
    """使用预处理仿射矩阵将关键点映射回原始图片坐标."""
    transform = np.asarray(sample["affine_transform"], dtype=np.float32)
    inverse = cv2.invertAffineTransform(transform)
    homogeneous = np.concatenate(
        (keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)),
        axis=1)
    return homogeneous @ inverse.T


def draw_result(image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                bbox: list[float], threshold: float) -> np.ndarray:
    """在原图上绘制身体骨架及所有有效 WholeBody 关键点."""
    result = image.copy()
    height, width = image.shape[:2]
    valid = ((scores >= threshold) &
             (keypoints[:, 0] >= 0) & (keypoints[:, 0] < width) &
             (keypoints[:, 1] >= 0) & (keypoints[:, 1] < height))
    x, y, box_width, box_height = bbox
    cv2.rectangle(
        result, (round(x), round(y)),
        (round(x + box_width), round(y + box_height)),
        (0, 255, 0), 2, cv2.LINE_AA)
    for first, second in BODY_SKELETON:
        if not valid[first] or not valid[second]:
            continue
        start = tuple(np.round(keypoints[first]).astype(int))
        end = tuple(np.round(keypoints[second]).astype(int))
        cv2.line(result, start, end, (255, 0, 255), 2, cv2.LINE_AA)
    for point, is_valid in zip(keypoints, valid):
        if not is_valid:
            continue
        center = tuple(np.round(point).astype(int))
        cv2.circle(result, center, 2, (0, 255, 255), -1, cv2.LINE_AA)
    return result


def postprocess(args: argparse.Namespace) -> None:
    """处理所有冒烟样本并验证两张图片的输出不同."""
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    outputs = metadata["outputs"]
    x_detail = next(item for item in outputs if item["shape"][-1] == 384)
    y_detail = next(item for item in outputs if item["shape"][-1] == 512)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    raw_signatures = []
    results = []
    for sample in metadata["samples"]:
        stem = sample["stem"]
        x_path = args.output_dir / f"{stem}_{x_detail['index']}.bin"
        y_path = args.output_dir / f"{stem}_{y_detail['index']}.bin"
        pred_x = load_output(x_path, x_detail)
        pred_y = load_output(y_path, y_detail)
        raw_signatures.append((pred_x.tobytes(), pred_y.tobytes()))
        keypoints, scores = decode_simcc(pred_x, pred_y)
        source_keypoints = map_to_source(keypoints, sample)
        image = cv2.imread(str(args.input_dir / sample["source_copy"]))
        if image is None:
            raise ValueError(f"无法读取 Demo 原图: {sample['source_copy']}")
        visualization = draw_result(
            image, source_keypoints, scores, sample["bbox_xywh"],
            args.score_threshold)
        output_image = args.result_dir / f"{stem}_keypoints.jpg"
        if not cv2.imwrite(str(output_image), visualization):
            raise ValueError(f"无法写入可视化: {output_image}")
        results.append({
            "stem": stem,
            "source_image": sample["source_image"],
            "annotation_id": sample["annotation_id"],
            "keypoints": [{
                "id": index,
                "x": float(point[0]),
                "y": float(point[1]),
                "score": float(scores[index]),
            } for index, point in enumerate(source_keypoints)],
            "visualization": output_image.name,
        })
        print(f"[POSTPROCESS] {stem}: 133 个关键点")
    if len(raw_signatures) >= 2 and raw_signatures[0] == raw_signatures[1]:
        raise RuntimeError("两张不同图片产生完全相同的原始输出, 疑似缓冲复用.")
    result_path = args.result_dir / "keypoints.json"
    result_path.write_text(
        json.dumps({"samples": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[OK] RTMPose 关键点结果: {result_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    postprocess(parse_args())
