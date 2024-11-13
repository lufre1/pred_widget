from typing import Callable, List, Optional, Sequence, Union
from napari.types import LayerData
from elf.io import open_file
import napari

PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    # If we recognize the format, we return the actual reader function
    if isinstance(path, str) and path.endswith(".mrc"):
        return elf_read_file
    # otherwise we return None.
    return None


def elf_read_file(path: PathOrPaths) -> List[LayerData]:
    try:
        with open_file(path, mode="r") as f:
        # data = {key: f[key][:] for key in f.keys()}
            data = f["data"][:]
        layer_attributes = {
            "name": "Raw",
            "colormap": "gray",
            "blending": "additive"
            }
        return [(data, layer_attributes)]
    except Exception as e:
        print(f"Failed to read file: {e}")
        return


# def main():
#     path = "/home/freckmann15/data/mitochondria/fidi_orig/20240722_M13DKO_1/37371_O5_66K_TS_SP_34-01_rec_2Kb1dawbp_crop_raw.mrc"
#     data = elf_read_file(path)
#     print(data.keys())


# if __name__ == "__main__":
#     main()