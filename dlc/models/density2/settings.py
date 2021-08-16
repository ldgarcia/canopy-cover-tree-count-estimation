from typing import Optional

from dlc.models.unet.settings import UNetSettings


class Density2Settings(UNetSettings):
    name: str = "density2"
    cover_threshold: float = 0.0
