from dataclasses import dataclass

@dataclass
class CaptureConfig:
    left: int
    top: int
    width: int
    height: int
    target_fps: int = 30
