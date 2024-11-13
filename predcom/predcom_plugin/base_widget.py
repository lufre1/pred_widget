import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSpinBox, QLineEdit, QGroupBox, QFormLayout, QFrame, QComboBox
from superqt import QCollapsible


class BaseWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # self.model = None
        self.viewer = napari.current_viewer()
        # layout = QVBoxLayout()
        
        # # Create the image selection dropdown
        # self.image_selector = QComboBox()
        
        # # Populate initial options
        # self.update_image_selector()
        
        # # Connect to Napari layer events to update the list
        # self.viewer.layers.events.inserted.connect(self.update_image_selector)
        # self.viewer.layers.events.removed.connect(self.update_image_selector)

        # # Add your buttons here
        # self.load_model_button = QPushButton('Load Model')
        # self.predict_button = QPushButton('Run Prediction')
        # self.settings = self._create_settings_widget()

        # # Connect buttons to functions
        # self.predict_button.clicked.connect(self.on_predict)
        # self.load_model_button.clicked.connect(self.on_load_model)

        # # Add the buttons to the layout
        # layout.addWidget(self.image_selector)
        # layout.addWidget(self.predict_button)
        # layout.addWidget(self.load_model_button)
        # layout.addWidget(self.settings)

        # self.setLayout(layout)

    def _add_int_param(self, name, value, min_val, max_val, title=None, step=1, layout=None, tooltip=None):
        if layout is None:
            layout = QHBoxLayout()
        label = QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)
        param = QSpinBox()
        param.setRange(min_val, max_val)
        param.setValue(value)
        param.setSingleStep(step)
        param.valueChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            param.setToolTip(tooltip)
        layout.addWidget(param)
        return param, layout

    def _add_choice_param(self, name, value, options, title=None, layout=None, update=None, tooltip=None):
        if layout is None:
            layout = QHBoxLayout()
        label = QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)

        # Create the dropdown menu via QComboBox, set the available values.
        dropdown = QComboBox()
        dropdown.addItems(options)
        if update is None:
            dropdown.currentIndexChanged.connect(lambda index: setattr(self, name, options[index]))
        else:
            dropdown.currentIndexChanged.connect(update)

        # Set the correct value for the value.
        dropdown.setCurrentIndex(dropdown.findText(value))

        if tooltip:
            dropdown.setToolTip(tooltip)

        layout.addWidget(dropdown)
        return dropdown, layout

    def _add_shape_param(self, names, values, min_val, max_val, step=1, title=None, tooltip=None):
        layout = QHBoxLayout()

        x_layout = QVBoxLayout()
        x_param, _ = self._add_int_param(
            names[0], values[0], min_val=min_val, max_val=max_val, layout=x_layout, step=step,
            title=title[0] if title is not None else title, tooltip=tooltip
        )
        layout.addLayout(x_layout)

        y_layout = QVBoxLayout()
        y_param, _ = self._add_int_param(
            names[1], values[1], min_val=min_val, max_val=max_val, layout=y_layout, step=step,
            title=title[1] if title is not None else title, tooltip=tooltip
        )
        layout.addLayout(y_layout)

        return x_param, y_param, layout

    def _make_collapsible(self, widget, title):
        parent_widget = QWidget()
        parent_widget.setLayout(QVBoxLayout())
        collapsible = QCollapsible(title, parent_widget)
        collapsible.addWidget(widget)
        parent_widget.layout().addWidget(collapsible)
        return parent_widget
