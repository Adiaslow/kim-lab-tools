"""Image eraser GUI implementation."""

import numpy as np
from qtpy.QtWidgets import (
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QPushButton,
    QVBoxLayout,
    QSlider,
    QLabel,
    QWidget,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap, QPainter, QColor

from ..utils.image_converter import numpy_to_qimage


class ImageEraser(QMainWindow):
    """GUI for erasing parts of an image."""

    closed = Signal()

    def __init__(self, image: np.ndarray):
        """Initialize the eraser.

        Args:
            image: Input image array.
        """
        super().__init__()
        self.image = image
        self.mask_image = np.zeros_like(self.image)
        self.drawing = False
        self.brush_size = 3
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Image Eraser")
        container = QWidget()
        ui_layout = QVBoxLayout()

        # Image view
        self.img_view = QGraphicsView(self)
        self.img_view.setMouseTracking(True)
        self.img_view.viewport().installEventFilter(self)

        self.img_scene = QGraphicsScene(self)
        self.qimg = numpy_to_qimage(self.image)
        self.img_pixmap = QPixmap.fromImage(self.qimg)
        self.img_scene.addPixmap(self.img_pixmap)
        self.img_view.setScene(self.img_scene)

        # Brush size control
        self.brush_size_slider = QSlider(Qt.Horizontal, self)
        self.brush_size_slider_label = QLabel("Brush Size")
        self.brush_size_slider_label.setAlignment(Qt.AlignmentFlag.AlignLeading)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(10)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)

        # Buttons
        self.save_button = QPushButton("Save", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.save_button.clicked.connect(self.save_mask)
        self.cancel_button.clicked.connect(self.cancel_changes)

        # Layout
        ui_layout.addWidget(self.img_view)
        ui_layout.addWidget(self.brush_size_slider_label)
        ui_layout.addWidget(self.brush_size_slider)
        ui_layout.addWidget(self.save_button)
        ui_layout.addWidget(self.cancel_button)
        container.setLayout(ui_layout)
        self.setCentralWidget(container)

    def eventFilter(self, source, event):
        """Handle mouse events for drawing."""
        if source is self.img_view.viewport():
            if event.type() == Qt.MouseMove and self.drawing:
                self.draw_on_image(event.pos())
                return True
            elif (
                event.type() == Qt.MouseButtonPress and event.button() == Qt.LeftButton
            ):
                self.drawing = True
                self.draw_on_image(event.pos())
                return True
            elif (
                event.type() == Qt.MouseButtonRelease
                and event.button() == Qt.LeftButton
            ):
                self.drawing = False
                return True
        return super().eventFilter(source, event)

    # ... rest of implementation ...
