"""Temporal stabilisation for the live detect-then-classify loop.

The classifier is stateless: it sees one crop and returns one distribution, with
no memory of the previous frame. On a webcam that means every hand tremor,
auto-exposure step and one-pixel box jitter produces a fresh, independent
prediction, and the displayed label flickers even though the object never
changed. Nothing here touches the model or its weights — this module only
remembers what the model already said and reports a consensus instead of the
latest sample.

Smoothing a *stream* of predictions is only meaningful per object. Averaging a
bottle's predictions together with a cardboard box's would be worse than no
smoothing at all, so predictions are grouped into tracks first: a detection is
matched to an existing track by box overlap (IoU) between consecutive frames,
and each track keeps its own rolling window of softmax vectors. With a single
item in front of the camera there is exactly one track and the machinery is
invisible; with several items each keeps its own independent history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class StableResult:
    """One track's smoothed verdict for the current frame."""

    track_id: int
    class_idx: int | None  # None -> below the confidence gate ("unknown")
    confidence: float  # window-averaged probability of the reported class
    votes: int  # frames in the window whose argmax agrees with the verdict
    window: int  # frames currently held (< history during warm-up)


@dataclass
class _Track:
    track_id: int
    box: Box
    history: deque[np.ndarray]
    last_frame: int
    # Currently displayed class, kept across frames so `switch_margin` has an
    # incumbent to defend. Deliberately NOT reset when the confidence gate
    # reports "unknown": a brief dip below threshold shouldn't let the label
    # jump to a different class on the way back up.
    shown_idx: int | None = None


class PredictionStabilizer:
    """Per-object rolling-window smoothing over classifier softmax outputs.

    Parameters
    ----------
    history:
        Frames kept per track. Larger = steadier but slower to react when the
        object is genuinely swapped. 10 at ~15 FPS is roughly a 0.7 s memory.
    mode:
        ``"mean"`` averages the softmax vectors and takes the argmax (uses the
        full distribution, so a run of "plastic 0.95" outweighs a single
        "paper 0.34"). ``"vote"`` is plain majority voting over per-frame
        argmaxes, which ignores how confident each frame was.
    conf_threshold:
        Smoothed confidence below this reports ``class_idx=None`` -> "unknown",
        unless the window is decisive — see ``min_agreement``.
    min_agreement:
        Escape hatch for the confidence gate. Averaging softmax vectors
        systematically *lowers* the peak: ten frames reading paper at
        68/32/65/76/57% average to ~56%, which a 0.60 gate would reject even
        though every frame agreed. So a verdict is also accepted when at least
        this fraction of the window picked it, regardless of the averaged
        probability. With 7 classes, chance agreement is ~14%, so 0.70 is far
        from accidental. "Unknown" then means what it should: the model is
        neither confident nor consistent. Set to 1.1 to disable and gate on
        probability alone.
    switch_margin:
        Hysteresis. A challenger must beat the currently displayed class by
        this much averaged probability before the label is allowed to change.
        Kills the residual two-class ping-pong when both sit near 0.45.
    iou_match:
        Minimum IoU to consider a detection the same object as an existing
        track. 0.3 is loose enough to survive a jittering box, tight enough
        not to merge two adjacent items.
    max_age:
        Frames a track survives without being matched, so a one-frame
        detector dropout doesn't discard the accumulated history.
    """

    def __init__(
        self,
        history: int = 10,
        mode: str = "mean",
        conf_threshold: float = 0.60,
        switch_margin: float = 0.05,
        iou_match: float = 0.30,
        max_age: int = 5,
        min_agreement: float = 0.70,
    ):
        if mode not in ("mean", "vote"):
            raise ValueError(f"mode must be 'mean' or 'vote', got {mode!r}")
        self.history = int(history)
        self.mode = mode
        self.conf_threshold = float(conf_threshold)
        self.min_agreement = float(min_agreement)
        self.switch_margin = float(switch_margin)
        self.iou_match = float(iou_match)
        self.max_age = int(max_age)
        self._tracks: list[_Track] = []
        self._next_id = 0

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def update(self, frame_idx: int, boxes: list[Box],
               probs: list[np.ndarray]) -> list[StableResult]:
        """Fold this frame's detections into their tracks, return the verdicts.

        ``boxes[i]`` and ``probs[i]`` describe the same detection. Results come
        back in the same order as the inputs.
        """
        # Drop tracks that haven't been seen recently.
        self._tracks = [t for t in self._tracks
                        if frame_idx - t.last_frame <= self.max_age]

        # Greedy IoU association: strongest overlaps claim their track first,
        # so a marginal 0.31 match can't steal a track from a 0.9 match.
        candidates = []
        for det_i, box in enumerate(boxes):
            for trk_i, track in enumerate(self._tracks):
                score = iou(box, track.box)
                if score >= self.iou_match:
                    candidates.append((score, det_i, trk_i))
        candidates.sort(reverse=True)

        det_to_track: dict[int, int] = {}
        claimed_dets: set[int] = set()
        claimed_tracks: set[int] = set()
        for _, det_i, trk_i in candidates:
            if det_i in claimed_dets or trk_i in claimed_tracks:
                continue
            det_to_track[det_i] = trk_i
            claimed_dets.add(det_i)
            claimed_tracks.add(trk_i)

        results: list[StableResult] = []
        for det_i, (box, prob) in enumerate(zip(boxes, probs)):
            if det_i in det_to_track:
                track = self._tracks[det_to_track[det_i]]
            else:
                track = _Track(self._next_id, box,
                               deque(maxlen=self.history), frame_idx)
                self._next_id += 1
                self._tracks.append(track)
            track.box = box
            track.last_frame = frame_idx
            track.history.append(np.asarray(prob, dtype=np.float64))
            results.append(self._decide(track))
        return results

    def _decide(self, track: _Track) -> StableResult:
        stacked = np.stack(track.history)  # [window, num_classes]
        mean_probs = stacked.mean(axis=0)
        per_frame_argmax = stacked.argmax(axis=1)

        if self.mode == "vote":
            counts = np.bincount(per_frame_argmax, minlength=stacked.shape[1])
            idx = int(counts.argmax())
        else:
            idx = int(mean_probs.argmax())

        # Hysteresis: defend the incumbent unless the challenger clears it by
        # `switch_margin`. Compared in averaged-probability space in both modes
        # so the margin means the same thing either way.
        if track.shown_idx is not None and idx != track.shown_idx:
            if mean_probs[idx] < mean_probs[track.shown_idx] + self.switch_margin:
                idx = track.shown_idx
        track.shown_idx = idx

        confidence = float(mean_probs[idx])
        votes = int((per_frame_argmax == idx).sum())
        agreement = votes / len(track.history)
        decisive = (confidence >= self.conf_threshold
                    or agreement >= self.min_agreement)
        gated = idx if decisive else None
        return StableResult(track.track_id, gated, confidence, votes,
                            len(track.history))