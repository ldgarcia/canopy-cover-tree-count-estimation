import abc
import dataclasses
import pathlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from dlc.tools.cache import BaseArrayCache


@dataclasses.dataclass
class FrameDataCreatorResult:
    """Define the result of a frame data creator.

    Parameters
    ----------
    tile_id
        The tile index.
    area_id
        The area index.
    key
        The key for the result (e.g. segmentation-mask).
    payload
        The derived data.
    payload_no_prefix_keys
        Keys in payload that should not be compounded with a prefix.
    """

    tile_id: int
    area_id: int
    key: str
    payload: Any
    payload_no_prefix_keys: Tuple[str, ...] = ()
    payload_main_key: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class FrameDataCreator(metaclass=abc.ABCMeta):
    """Define interface of a frame data creator.

    Notes
    -----
    A frame data creator produces derived data for a particular frame \
        which is defined as the intersection of a tile and an area.
    """

    _key: str

    @property
    def key(self) -> str:
        return self._key

    @abc.abstractmethod
    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        overwrite: bool = True,
        verbose: bool = True,
        cache: Optional[BaseArrayCache] = None,
    ) -> FrameDataCreatorResult:
        """Run the frame creator.

        Parameters
        ----------
        area_id
            The index of the area in the database.
        tile_id: optional
            The index of the tile in the database.
        verbose
            Print diagnostic information.

        Returns
        -------
        FrameDataCreatorResult
            The result.
        """

    def _cache_key(self, ns: str, polygon_id: int) -> str:
        return f"{ns}:{polygon_id}"

    def _compose_key(self, parts: List[str]) -> str:
        return "-".join(parts).replace(".", "")


class RasterFrameDataCreator(FrameDataCreator):
    _only_filename: bool
    _output_base_path: pathlib.Path
    _file_count: int = 1

    @property
    def only_filename(self) -> bool:
        return self._only_filename

    @only_filename.setter
    def only_filename(self, value: bool) -> None:
        self._only_filename = value

    @property
    def output_base_path(self) -> pathlib.Path:
        return self._output_base_path

    def _check_output_base_path(self) -> None:
        if not self._output_base_path.exists():
            self._output_base_path.mkdir(parents=True)

    def output_path(self, area_id: int, tile_id: int) -> pathlib.Path:
        name = f"{area_id}-{tile_id}-{self._key}.tif"
        return self._output_base_path.joinpath(name)

    def output_filename(self, output_path: pathlib.Path) -> str:
        if self._only_filename:
            return output_path.name
        return str(output_path)
