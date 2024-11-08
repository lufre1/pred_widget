from typing import TYPE_CHECKING
import h5py
from magicgui import magic_factory, widgets
import napari
from napari.utils.notifications import show_info
from napari import Viewer
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit
from elf.io import open_file
# import napari

# Custom imports for model and prediction utilities
from ..util import get_model, run_prediction

# if TYPE_CHECKING:
#     import napari


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.image = None
        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()
        
        # Tile shape input
        tile_shape_layout = QHBoxLayout()
        tile_shape_label = QLabel("Tile Shape:")
        self.tile_shape_input = QLineEdit("(32, 256, 256)")
        tile_shape_layout.addWidget(tile_shape_label)
        tile_shape_layout.addWidget(self.tile_shape_input)
        layout.addLayout(tile_shape_layout)

        # Halo input
        halo_layout = QHBoxLayout()
        halo_label = QLabel("Halo:")
        self.halo_input = QLineEdit("(4, 32, 32)")
        halo_layout.addWidget(halo_label)
        halo_layout.addWidget(self.halo_input)
        layout.addLayout(halo_layout)

        # Add your buttons here
        self.load_model_button = QPushButton('Load Model')
        self.predict_button = QPushButton('Run Prediction')
        self.load_image_button = QPushButton('Load Image')

        # Connect buttons to functions
        self.predict_button.clicked.connect(self.on_predict)
        self.load_image_button.clicked.connect(self.on_load_image)
        self.load_model_button.clicked.connect(self.on_load_model)

        # Add the buttons to the layout
        layout.addWidget(self.predict_button)
        layout.addWidget(self.load_image_button)
        layout.addWidget(self.load_model_button)

        self.setLayout(layout)

    def on_load_model(self):
        # Open file dialog to select a model
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Model (*.pt)")
        file_dialog.setViewMode(QFileDialog.List)

        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                # Assuming you load a single model path here
                model_path = file_paths[0]
                self.load_model(model_path)

    def load_model(self, model_path):
        print("model path type and value", type(model_path), model_path)
        # Load the model from the selected path
        model = get_model(model_path)
        self.model = model
    
    def on_predict(self):
        # Get the model and postprocessing settings.
        model = self.model
        if model is None:
            show_info("Model not loaded.")
            return
        if self.image is None:
            show_info("Image not loaded.")
            return
        
        # Get image from the viewer
        image = np.asarray(self.viewer.layers[0].data)
        # get tile shape and halo from the viewer
        tile_shape = eval(self.tile_shape_input.text())
        halo = eval(self.halo_input.text())
        predictions = run_prediction(image, model, block_shape=tile_shape, halo=halo)
        # Add predictions to Napari as separate layers
        for i, pred in enumerate(predictions):
            layer_name = f"Prediction {i+1}"
            self.viewer.add_image(pred, name=layer_name, colormap="inferno", blending="additive")

    def on_load_image(self):
        # Open file dialog to select an image
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.tif *.png *.jpg *.bmp *.h5 *.z5 *.nrrd)")
        file_dialog.setViewMode(QFileDialog.List)

        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                # Assuming you load a single image path here
                image_path = file_paths[0]
                self.load_image(image_path)
                self.image = image_path

    def load_image(self, image_path):
        print("image path type and value", type(image_path), image_path)
        # Load the image from the selected path and display it in Napari
        image = open_file(image_path, mode="r")
        
        for k, v in image.items():
            if "mito" in k:
                self.viewer.add_labels(v[::2, ::2, ::2], name=k)
            elif "raw" in k:
                self.viewer.add_image(v[::2, ::2, ::2], name=k)
            # else:
            #     self.viewer.add_image(v, name=k)
            # viewer = napari.view_image(image["raw"])
        # viewer.title = f"Loaded: {image_path}"


def predict_widget():
    return MyWidget()


# @magic_factory(call_button="Predict", model_path={"label": "Model Path"})
# def predict_widget(
#     viewer: "napari.Viewer",
#     image: "napari.layers.Image",
#     model_path: str = "/home/freckmann15/data/mitochondria/models/mito",
#     tile_shape: tuple[int, int, int] = (32, 256, 256),
#     halo: tuple[int, int, int] = (4, 32, 32),
# ) -> None:
#     """Predict with a model on an image and add the result as a new layer."""

#     # Load the model
#     model = get_model(model_path)
#     if model is None:
#         show_info(f"Model not found at: {model_path}")
#         return
#     print("model", model)
#     print("model type", type(model))

#     # Convert image to numpy array
#     image_data = np.asarray(image.data, dtype=np.float32)
#     with h5py.File("/home/freckmann15/data/mitochondria/cooper/new_mitos/01_hoi_maus_2020_incomplete/A_WT_SC_DIV14/WT_Unt_SC_09175_B5_01_DIV16_mtk_01.h5", "r") as f:
#         image_data = np.array(f["raw"], dtype=np.float32)
#     print("image shape and type", image_data.shape, image_data.dtype)

#     # Run prediction
#     try:
#         # breakpoint()
#         predictions = run_prediction(image_data, model, block_shape=tile_shape, halo=halo)
#     except Exception as e:
#         show_info(f"Prediction error: {e}")
#         print(e)
#         return

#     # Add predictions to Napari as separate layers
#     for i, pred in enumerate(predictions):
#         layer_name = f"Prediction {i+1}"
#         viewer.add_image(pred, name=layer_name, colormap="inferno", blending="additive")

# Add additional widgets for parameters if needed
# model_path_widget = widgets.FileEdit(value="/path/to/your/model/checkpoint")
# tile_shape_widget = widgets.Tuple(default=(32, 256, 256), label="Tile Shape")
# halo_widget = widgets.Tuple(default=(4, 32, 32), label="Halo")

# Optional: Assemble into a more custom layout
# viewer.window.add_dock_widget(predict_widget, area="right")
