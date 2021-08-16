from dlc.models.base.settings import Settings


class UNet0Settings(Settings):
    name: str = "unet0"
    layer_count: int = 64
    output_name: str = "segmentation_map"
