from typing import Literal, TYPE_CHECKING, Tuple
import h5py
from magicgui import magic_factory
from napari.utils.notifications import show_info
import numpy as np
# from ._widgets import EmbeddingWidget

from ..util import get_model, run_prediction
# from ._widgets import EmbeddingWidget

if TYPE_CHECKING:
    import napari


# @magic_factory(call_button="Predict")
# def main(
#     image: "napari.layers.Image",
#     model_path: str = "/home/freckmann15/data/mitochondria/models/mito",
#     tile_shape: tuple[int, int, int] = (32, 256, 256),
#     halo: tuple[int, int, int] = (4, 32, 32),
#     # min_distance: int = 10,
#     # threshold_abs: float = 1.0,
# ) -> "napari.types.LayerDataTuple":
#     # if viewer is None:
#     #     viewer = napari.Viewer()
#     # viewer.add
    
#     # Get the model and postprocessing settings.
#     model_ = get_model(model_path)
#     if model_ is None:
#         show_info(f"Model checkpoint not found at {model_path}.")
#         return
#     # min_distance, threshold_abs = get_postprocessing_parameters(model)

#     image_data = np.array(image.data, dtype=np.float32)
#     pred = run_prediction(model_, image_data)
#     layers = []
#     for i, prediction in enumerate(pred):
#         layer = (prediction, {"name": f"Prediction {i+1}", "colormap": "inferno", "blending": "additive"})
#         layers.append(layer)

#     return layers


@magic_factory(call_button="Predict", model_path={"label": "Model Path"})
def predict_widget(
    viewer: "napari.Viewer",
    image: "napari.layers.Image",
    model_path: str = "/home/freckmann15/data/mitochondria/models/mito",
    tile_shape: tuple[int, int, int] = (32, 256, 256),
    halo: tuple[int, int, int] = (4, 32, 32),
) -> None:
    """Predict with a model on an image and add the result as a new layer."""

    # Load the model
    model = get_model(model_path)
    if model is None:
        show_info(f"Model not found at: {model_path}")
        return
    print("model", model)
    print("model type", type(model))

    # Convert image to numpy array
    image_data = np.asarray(image.data, dtype=np.float32)
    with h5py.File("/home/freckmann15/data/mitochondria/cooper/new_mitos/01_hoi_maus_2020_incomplete/A_WT_SC_DIV14/WT_Unt_SC_09175_B5_01_DIV16_mtk_01.h5", "r") as f:
        image_data = np.array(f["raw"], dtype=np.float32)
    print("image shape and type", image_data.shape, image_data.dtype)

    # Run prediction
    try:
        # breakpoint()
        predictions = run_prediction(image_data, model, block_shape=tile_shape, halo=halo)
    except Exception as e:
        show_info(f"Prediction error: {e}")
        print(e)
        return

    # Add predictions to Napari as separate layers
    for i, pred in enumerate(predictions):
        layer_name = f"Prediction {i+1}"
        viewer.add_image(pred, name=layer_name, colormap="inferno", blending="additive")

# @magic_factory(call_button="Predict")
# def main(
#     viewer: "napari.viewer.Viewer",
#     image: "napari.layers.Image",
#     model_path: str = "/home/freckmann15/data/mitochondria/models/mito",
#     tile_shape: Tuple[int, int, int] = (32, 256, 256),
#     halo: Tuple[int, int, int] = (4, 32, 32),
# ) -> "napari.types.LayerDataTuple":

#     model = get_model(model_path)
#     if model is None:
#         show_info(f"Model checkpoint not found at {model_path}.")
#         return

#     # Convert image data for prediction
#     image_data = np.array(image.data, dtype=np.float32)
#     try:
#         predictions = run_prediction(image_data, model, block_shape=tile_shape, halo=halo)
#     except Exception as e:
#         show_info(f"Prediction error: {e}")
#         return

#     # Generate and add prediction layers
#     layers = []
#     for i, prediction in enumerate(predictions):
#         layer = (prediction, {"name": f"Prediction {i+1}", "colormap": "inferno", "blending": "additive"})
#         layers.append(layer)
#         viewer.add_image(prediction, name=f"Prediction {i+1}", colormap="inferno", blending="additive")
    
#     return layers


    # layers = []
    # return layers
    # points = run_counting(model_, image_data, min_distance=min_distance, threshold_abs=threshold_abs)
    # count = len(points)

    # Set the size of the points dependend on the size of the image.
    # image_shape = image_data.shape if image_data.ndim == 2 else image_data.shape[:-1]
    # if any(sh > 2048 for sh in image_shape):
    #     point_size = 20
    # else:
    #     point_size = 10
    # layer_kwargs = {
    #     "name": "Counting Result",
    #     "size": point_size,
    # }

    # show_info(f"STACC counted {count} {model}.")
    # return points, layer_kwargs, "points"


# if __name__ == "__main__":
#     main()