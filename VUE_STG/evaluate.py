import os
import os.path as osp
from typing import Optional, Any, Literal

import numpy as np
import pandas as pd

from tube import Tube

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1
Slice = list[BBox]


def _union_area(rects: list[BBox]) -> float:
    """
    Compute the union area of axis-aligned rectangles.

    Approach:
        - Collect all unique x-coordinates (rectangle vertical edges).
        - Sweep adjacent x-intervals to form vertical strips.
        - For each strip, gather y-intervals covered by any rectangle overlapping the strip.
        - Merge y-intervals, accumulate covered length, then multiply by strip width.

    Complexity:
        Roughly O(Ux * K log K), where:
            Ux = number of unique x-intervals
            K  = number of active rectangles per strip
    """
    if not rects:
        return 0.0

    xs = sorted({x for x0, _, x1, _ in rects for x in (x0, x1)})
    area = 0.0

    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        dx = x1 - x0
        if dx <= 0:
            continue

        # Collect y-intervals of rectangles that overlap this x-strip
        intervals: list[tuple[float, float]] = []
        for rx0, ry0, rx1, ry1 in rects:
            if not (rx1 <= x0 or rx0 >= x1):  # overlaps the strip
                intervals.append((ry0, ry1))
        if not intervals:
            continue

        # Merge y-intervals
        intervals.sort()
        covered = 0.0
        cy0, cy1 = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 > cy1:
                # disjoint interval
                covered += cy1 - cy0
                cy0, cy1 = y0, y1
            else:
                # overlapping interval, extend current segment
                if y1 > cy1:
                    cy1 = y1
        covered += cy1 - cy0

        area += covered * dx

    return area


def _pairwise_intersections(a_rects: list[BBox], b_rects: list[BBox]) -> list[BBox]:
    """
    Generate all pairwise intersection rectangles (with positive area)
    between rectangles in A and B.
    """
    inter_rects: list[BBox] = []
    for ax0, ay0, ax1, ay1 in a_rects:
        for bx0, by0, bx1, by1 in b_rects:
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            if ix1 > ix0 and iy1 > iy0:
                inter_rects.append((ix0, iy0, ix1, iy1))
    return inter_rects


def compute_region_inter_union(
    slice_a: list[BBox],
    slice_b: list[BBox],
) -> tuple[float, float, float, float]:
    """
    Compute intersection/union info for two regions represented by unions of rectangles.

    Args:
        slice_a: List of rectangles representing region A (union of all).
        slice_b: List of rectangles representing region B (union of all).

    Returns:
        (inter, union, area_a, area_b):
            inter  : intersection area of region A and region B
            union  : union area of region A and region B
            area_a : union area of region A
            area_b : union area of region B

    Notes:
        Intersection of unions:
            (⋃A) ∩ (⋃B) = ⋃_{a∈A, b∈B} (a ∩ b)

        We enumerate all pairwise intersection rectangles, then compute their union area.
    """
    area_a = _union_area(slice_a)
    area_b = _union_area(slice_b)

    if not slice_a or not slice_b:
        inter = 0.0
        union = area_a + area_b
        return inter, union, area_a, area_b

    inter_rects = _pairwise_intersections(slice_a, slice_b)
    inter = _union_area(inter_rects)

    union = area_a + area_b - inter
    return inter, union, area_a, area_b


def compute_inter_union(
    bbox_a: Optional[BBox],
    bbox_b: Optional[BBox],
) -> tuple[float, float, float, float]:
    """
    Compute intersection/union for two single bounding boxes (or None).

    Args:
        bbox_a: (x0, y0, x1, y1) or None.
        bbox_b: (x0, y0, x1, y1) or None.

    Returns:
        (inter, union, area_a, area_b)
    """
    if bbox_a is None:
        area_a = 0.0
    else:
        ax0, ay0, ax1, ay1 = bbox_a
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)

    if bbox_b is None:
        area_b = 0.0
    else:
        bx0, by0, bx1, by1 = bbox_b
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    if bbox_a is None or bbox_b is None:
        inter = 0.0
        union = area_a + area_b
        return inter, union, area_a, area_b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter = inter_w * inter_h

    union = area_a + area_b - inter
    return inter, union, area_a, area_b


def _rename_metric_keys(raw_result: dict[str, float]) -> dict[str, float]:
    """
    Map legacy key names (ioa/iob/iou) to final metric names
    (recall/precision/iou). Keeps all other keys unchanged.
    """
    mapped: dict[str, float] = {}
    for key, value in raw_result.items():
        if "ioa" in key:
            new_key = key.replace("ioa", "recall")
        elif "iob" in key:
            new_key = key.replace("iob", "precision")
        elif "iou" in key:
            new_key = key.replace("iou", "iou")
        else:
            new_key = key
        mapped[new_key] = value
    return mapped


def compare_tubes(
    tube_a: Tube,
    tube_b: Tube,
    multi_boxes_policy: Literal["first", "last", "union"] = "first",
) -> dict[str, Any]:
    """
    Compare two spatio-temporal tubes frame by frame and compute metrics.

    Args:
        tube_a: Ground-truth or reference tube.
        tube_b: Prediction or comparison tube.
        multi_boxes_policy:
            - "first": use the first bbox in each frame
            - "last" : use the last bbox in each frame
            - "union": use union of all bboxes in each frame

    Returns:
        Dict with time-based and volume-based metrics (IoU, recall, precision),
        including both 3D metrics and legacy aggregated 2D metrics.
    """
    inter_list: list[float] = []
    union_list: list[float] = []
    area_a_list: list[float] = []
    area_b_list: list[float] = []

    # union of frame indices
    ts = tube_a.slices.keys() | tube_b.slices.keys()
    assert ts, "Both tubes are empty; there is no frame to compare."

    for t in ts:
        slice_a: Slice = tube_a.slices.get(t, [])
        slice_b: Slice = tube_b.slices.get(t, [])

        if multi_boxes_policy == "first":
            slice_a, slice_b = slice_a[:1], slice_b[:1]
        elif multi_boxes_policy == "last":
            slice_a, slice_b = slice_a[-1:], slice_b[-1:]
        # "union" means keeping all boxes and using region-union logic

        inter, union, area_a, area_b = compute_region_inter_union(slice_a, slice_b)

        # Sanity check for single-box case
        if len(slice_a) == 1 and len(slice_b) == 1:
            bbox_a = slice_a[0]
            bbox_b = slice_b[0]
            _inter, _union, _area_a, _area_b = compute_inter_union(bbox_a, bbox_b)
            assert np.isclose(inter, _inter)
            assert np.isclose(union, _union)
            assert np.isclose(area_a, _area_a)
            assert np.isclose(area_b, _area_b)

        inter_list.append(inter)
        union_list.append(union)
        area_a_list.append(area_a)
        area_b_list.append(area_b)

    inter_arr = np.asarray(inter_list, dtype=float)
    union_arr = np.asarray(union_list, dtype=float)
    area_a_arr = np.asarray(area_a_list, dtype=float)
    area_b_arr = np.asarray(area_b_list, dtype=float)

    num_a_frames = (area_a_arr > 0).sum()
    num_b_frames = (area_b_arr > 0).sum()
    num_inter_frames = np.logical_and(area_a_arr > 0, area_b_arr > 0).sum()
    num_union_frames = np.logical_or(area_a_arr > 0, area_b_arr > 0).sum()

    eps = np.finfo(float).eps
    iou_2d_list = inter_arr / (union_arr + eps)
    iou_2d_sum = iou_2d_list.sum()

    # Temporal level metrics (frame-based)
    t_iou = None if num_union_frames == 0 else num_inter_frames / num_union_frames
    t_ioa = None if num_a_frames == 0 else num_inter_frames / num_a_frames
    t_iob = None if num_b_frames == 0 else num_inter_frames / num_b_frames

    # Volume level metrics (region volume across time)
    v_iou_3d = None if num_union_frames == 0 else inter_arr.sum() / (union_arr.sum() + eps)
    v_ioa_3d = None if num_a_frames == 0 else inter_arr.sum() / (area_a_arr.sum() + eps)
    v_iob_3d = None if num_b_frames == 0 else inter_arr.sum() / (area_b_arr.sum() + eps)

    # Legacy (mean 2D IoU) metrics
    legacy_v_iou = None if num_union_frames == 0 else iou_2d_sum / num_union_frames
    legacy_v_ioa = None if num_a_frames == 0 else iou_2d_sum / num_a_frames
    legacy_v_iob = None if num_b_frames == 0 else iou_2d_sum / num_b_frames
    legacy_v_iou_intersection = None if num_inter_frames == 0 else iou_2d_sum / num_inter_frames

    return {
        "t_iou": t_iou,
        "t_ioa": t_ioa,
        "t_iob": t_iob,
        "v_iou_3d": v_iou_3d,
        "v_ioa_3d": v_ioa_3d,
        "v_iob_3d": v_iob_3d,
        "legacy_v_iou": legacy_v_iou,
        "legacy_v_ioa": legacy_v_ioa,
        "legacy_v_iob": legacy_v_iob,
        "legacy_v_iou_intersection": legacy_v_iou_intersection,
    }


class SpatioTemporalEvaluator:
    """
    Evaluator for spatio-temporal tubes with support for:
        - loading dataset metadata
        - loading ground-truth & prediction tubes
        - computing per-query metrics and grouped statistics
    """

    def __init__(self, step_ms: int = 1000):
        self.step_ms = step_ms
        self.video_info_dict: dict[str, dict] = {}
        self.query_info_dict: dict[str, dict] = {}
        self.gt_tube_dict: dict[str, Tube] = {}

    def load_dataset(self, path: str = "vue-stg-benchmark") -> None:
        """
        Load video info, query info and ground truth tubes
        from a given dataset root directory.
        """
        self.load_video_info(osp.join(path, "video.csv"))
        self.load_query_info(osp.join(path, "query.csv"))
        self.load_gt_tubes(osp.join(path, "tubes.csv"))

    def _assert_csv(self, file: str) -> None:
        assert file.endswith(".csv"), f"Expected a CSV file, got: {file}"

    def load_video_info(self, file: str) -> None:
        self._assert_csv(file)
        df = pd.read_csv(file)
        video_info = df.to_dict(orient="records")
        self.video_info_dict = {item["video_id"]: item for item in video_info}

    def load_query_info(self, file: str) -> None:
        self._assert_csv(file)
        df = pd.read_csv(file)
        query_info = df.to_dict(orient="records")
        self.query_info_dict = {item["query_id"]: item for item in query_info}

    def load_gt_tubes(self, file: str) -> None:
        self._assert_csv(file)
        self.gt_tube_dict = Tube.load_tubes_from_csv(file, self.step_ms)

    def load_pred_tubes(self, file: str) -> dict[str, Tube]:
        self._assert_csv(file)
        return Tube.load_tubes_from_csv(file, self.step_ms)

    def _evaluate_single(
        self,
        qid: str,
        gt_tube: Tube,
        pred_tube: Tube,
        video_length: float,
    ) -> dict[str, Any]:
        raw_result = compare_tubes(gt_tube, pred_tube)
        avg_area = gt_tube.get_avg_area()
        gt_length = gt_tube.get_length()

        result = {
            "query_id": qid,
            "avg_area": avg_area,
            "video_length": video_length,
            "gt_length": gt_length,
        }
        result.update(_rename_metric_keys(raw_result))
        return result

    def evaluate_pred_file(
        self,
        file: str,
        grouped: bool = True,
        ignore_missing_pred: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate a prediction CSV file against the loaded ground-truth tubes.

        Args:
            file: CSV file containing predicted tubes.
            grouped:
                If True, add area/video_length/gt_length group columns.
            ignore_missing_pred:
                If True, skip queries that have no prediction.
                If False, use empty tubes as prediction for missing queries.

        Returns:
            DataFrame with one row per query and all metrics.
        """
        pred_tube_dict = self.load_pred_tubes(file)

        print(f"Num pred tubes  : {len(pred_tube_dict)}")

        results: list[dict[str, Any]] = []
        for qid, gt_tube in self.gt_tube_dict.items():
            if qid in pred_tube_dict:
                pred_tube = pred_tube_dict[qid]
            elif ignore_missing_pred:
                continue
            else:
                pred_tube = Tube.empty_tube(step_ms=self.step_ms)

            vid = self.query_info_dict[qid]["video_id"]
            video_duration = self.video_info_dict[vid]["video_duration"]

            results.append(
                self._evaluate_single(
                    qid=qid,
                    gt_tube=gt_tube,
                    pred_tube=pred_tube,
                    video_length=video_duration,
                )
            )

        df = pd.DataFrame(results)
        col_map = {
            "t_iou": "t_IoU",
            "t_recall": "t_Recall",
            "t_precision": "t_Precision",
            "v_iou_3d": "3D_IoU",
            "v_recall_3d": "3D_Recall",
            "v_precision_3d": "3D_Precision",
            "legacy_v_iou": "v_IoU",
            "legacy_v_recall": "v_Recall",
            "legacy_v_precision": "v_Precision",
            "legacy_v_iou_intersection": "v_IoU_Int",
        }
        df = df.rename(columns=col_map)

        if grouped:
            df = self.add_groups(df)

        return df

    @staticmethod
    def add_groups(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add discretized group columns based on:
            - avg_area          -> area_group
            - video_length      -> video_length_group
            - gt_length         -> gt_length_group
        """
        df = df.copy()

        area_bins = [-float("inf"), 0.10, 0.30, float("inf")]
        area_labels = ["<10%", "10%-30%", ">30%"]

        video_length_bins = [-float("inf"), 60, 600, 1800]
        video_length_labels = ["<1min", "1-10min", "10-30min"]

        gt_length_bins = [-float("inf"), 3, 10, 60]
        gt_length_labels = ["<3s", "3-10s", "10-60s"]

        df["area_group"] = pd.cut(
            df["avg_area"], bins=area_bins, labels=area_labels, right=False
        )
        df["video_length_group"] = pd.cut(
            df["video_length"], bins=video_length_bins, labels=video_length_labels, right=False
        )
        df["gt_length_group"] = pd.cut(
            df["gt_length"], bins=gt_length_bins, labels=gt_length_labels, right=False
        )

        return df

def summarize_and_save_eval_simple(
    df: pd.DataFrame,
    out_dir: str,
) -> str:
    import os, os.path as osp
    os.makedirs(out_dir, exist_ok=True)

    # Ensure group columns exist
    need_group_cols = any(
        col not in df.columns
        for col in ["area_group", "video_length_group", "gt_length_group"]
    )
    if need_group_cols:
        df = SpatioTemporalEvaluator.add_groups(df)

    # ---- Compute group-wise means ----
    def gmean(col):
        return (
            df.groupby(col, dropna=False, observed=True)
            .mean(numeric_only=True)
            .reset_index()
        )

    # Overall mean
    overall = df.mean(numeric_only=True).to_frame().T
    overall["category"] = "overall"
    overall["group"] = "overall"

    # Area group mean
    area = gmean("area_group").rename(columns={"area_group": "category"})
    area["group"] = "object size"

    # Video length group mean
    vlen = gmean("video_length_group").rename(columns={"video_length_group": "category"})
    vlen["group"] = "video duration"

    # Ground truth length group mean
    gtlen = gmean("gt_length_group").rename(columns={"gt_length_group": "category"})
    gtlen["group"] = "gt duration"

    # ---- Merge all summaries ----
    df_all = pd.concat([overall, area, vlen, gtlen], ignore_index=True)

    # ---- Select the desired columns ----
    metrics = [
        "t_Precision", "t_Recall", "t_IoU",
        "v_Precision", "v_Recall", "v_IoU", "v_IoU_Int",
    ]
    cols = ["group", "category"] + [m for m in metrics if m in df_all.columns]

    summary = df_all[cols]

    print(summary.round(4))

    # ---- Save summary.csv ----
    out_path = osp.join(out_dir, "summary.csv")
    summary.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    result_paths = {
        'vidi2': 'results/vidi2/tubes.csv',
        'gemini-3-pro-preview': 'results/gemini-3-pro-preview/tubes.csv',
        'gpt-5': 'results/gpt-5/tubes.csv',
        'qwen3-vl-32b-instruct': 'results/qwen3-vl-32b-instruct/tubes.csv',
    }

    evaluator = SpatioTemporalEvaluator(step_ms=1000)
    evaluator.load_dataset("vue-stg-benchmark")

    for model_name, result_path in result_paths.items():
        print('=' * 64)
        print(f'Model: {model_name}')
        df = evaluator.evaluate_pred_file(result_path, ignore_missing_pred=False)
        summarize_and_save_eval_simple(df, f"output/{model_name}")
