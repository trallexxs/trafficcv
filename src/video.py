"""Video probing and fast sampled decoding.

Decoding high-resolution footage is the dominant cost, so frames are decoded
by a bundled ffmpeg (imageio-ffmpeg, works offline) which drops unselected
frames and downsizes the kept ones before they reach Python. OpenCV is the
fallback if ffmpeg cannot be started.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def probe(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 25.0
    meta = VideoMeta(path, float(fps), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return meta


def output_size(meta: VideoMeta, width: int) -> Tuple[int, int]:
    width = min(width, meta.width)
    height = int(round(meta.height * width / meta.width / 2.0)) * 2
    return width - width % 2, height


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def iter_frames(meta: VideoMeta, stride: int, width: int) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_sec, bgr_frame) for every `stride`-th frame, resized to `width`."""
    exe = _ffmpeg_exe()
    if exe is not None:
        try:
            yield from _iter_ffmpeg(exe, meta, stride, width)
            return
        except _FFmpegFailed:
            pass
    yield from _iter_opencv(meta, stride, width)


class _FFmpegFailed(RuntimeError):
    pass


def _iter_ffmpeg(exe: str, meta: VideoMeta, stride: int, width: int):
    w, h = output_size(meta, width)
    vf = f"select='not(mod(n\\,{stride}))',scale={w}:{h}:flags=area"
    cmd = [exe, "-v", "error", "-nostdin", "-threads", "0", "-i", meta.path,
           "-vf", vf, "-fps_mode", "passthrough", "-an", "-sn",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=w * h * 3 * 4)
    nbytes = w * h * 3
    k = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            idx = k * stride
            yield idx, idx / meta.fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if k == 0:
        raise _FFmpegFailed(meta.path)


def _iter_opencv(meta: VideoMeta, stride: int, width: int):
    w, h = output_size(meta, width)
    cap = cv2.VideoCapture(meta.path)
    idx = 0
    while True:
        if idx % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            yield idx, idx / meta.fps, frame
        elif not cap.grab():
            break
        idx += 1
    cap.release()
