from typing import Optional
import pandas as pd

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1
Slice = list[BBox]


def _sanitize_bbox(b: BBox) -> BBox:
    x0, y0, x1, y1 = b
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    return (x0, y0, x1, y1)


def quantize_time_ms(timestamp_ms: int, step_ms: int = 1000) -> int:
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    return ((timestamp_ms * 2 + step_ms) // (2 * step_ms)) * step_ms


class Tube:
    def __init__(self, step_ms: int):
        self.step_ms = step_ms
        self.slices = {}

    @classmethod
    def empty_tube(cls, step_ms: int) -> "Tube":
        return cls(step_ms)

    def get_avg_area(self):
        area_list = []
        for t, bboxes in self.slices.items():
            for bbox in bboxes:
                x0, y0, x1, y1 = bbox
                area = (x1 - x0) * (y1 - y0)
                area_list.append(area)
        return sum(area_list) / len(area_list) if area_list else 0.0
    
    def get_length(self):  # number of timestamps
        length = 0
        for key, value in self.slices.items():
            if len(value) > 0:
                length += 1
        return length

    def add_bbox(self, timestamp_ms: int, bbox: BBox):
        t_quantized = quantize_time_ms(timestamp_ms, self.step_ms)
        if t_quantized not in self.slices:
            self.slices[t_quantized] = []
        self.slices[t_quantized].append(_sanitize_bbox(bbox))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, step_ms: int) -> "Tube":
        required = ["time_ms", "x0", "y0", "x1", "y1"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing required columns (exact match): {', '.join(missing)}"
            )

        tube = Tube(step_ms)
        sub = df[required].copy()
        for row in sub.itertuples(index=False):
            time_ms = int(row[0])
            x0 = float(row[1])
            y0 = float(row[2])
            x1 = float(row[3])
            y1 = float(row[4])
            bbox = (x0, y0, x1, y1)
            tube.add_bbox(time_ms, bbox)

        return tube

    @classmethod
    def load_tubes_from_csv(cls, csv_file, step_ms: int) -> dict[str, "Tube"]:
        df = pd.read_csv(csv_file)
        required = ["query_id", "time_ms", "x0", "y0", "x1", "y1"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing required columns (exact match): {', '.join(missing)}"
            )

        df = df.dropna(subset=required).copy()
        df["time_ms"] = df["time_ms"].astype(int)

        result: dict[str, Tube] = {}

        for qid, g in df.groupby("query_id", sort=False):
            sub = g[["time_ms", "x0", "y0", "x1", "y1"]].copy()
            tube = cls.from_dataframe(sub, step_ms)
            result[qid] = tube

        return result

    def to_single_frame_tubes(self) -> dict[int, "Tube"]:
        single_frame_tubes = {}
        for t, slice in self.slices.items():
            new_tube = Tube(self.step_ms)
            new_tube.slices[t] = slice
            single_frame_tubes[t] = new_tube
        return single_frame_tubes
