"""Registration visualization GUI."""

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
from pathlib import Path
import nrrd
from typing import List, Tuple

from ..utils.slice_utils import slice_3d_volume


class AtlasSliceViewer:
    """GUI for viewing and manipulating atlas slices."""

    def __init__(self, atlas_path: Path):
        """Initialize viewer.

        Args:
            atlas_path: Path to atlas file.
        """
        self.atlas_path = Path(atlas_path).expanduser()
        self.atlas, _ = nrrd.read(self.atlas_path)
        self.active_slider = 0
        self.x_angle = 0
        self.y_angle = 0
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize user interface."""
        self.root = tk.Tk()
        self.root.title("Atlas Viewer")

        # Create sliders
        max_dims = [self.atlas.shape[i] - 1 for i in range(3)]
        self.sliders = self._create_sliders(max_dims)

        # Create figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self._get_current_slice(), cmap="gray")

        # Add save button
        self.btn_save = tk.Button(
            self.root, text="Save as PNG", command=self.save_image
        )
        self.btn_save.pack()

        # Add canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

    def _create_sliders(self, max_dims: List[int]) -> List[tk.Scale]:
        """Create UI sliders.

        Args:
            max_dims: Maximum value for each dimension.

        Returns:
            List of slider widgets.
        """
        sliders = []

        # Position sliders
        for i in range(3):
            slider = tk.Scale(
                self.root,
                from_=0,
                to=max_dims[i],
                orient=tk.HORIZONTAL,
                command=lambda v, i=i: self.update_slice(i),
            )
            slider.pack()
            sliders.append(slider)

        # Angle sliders
        for _ in range(2):
            slider = tk.Scale(
                self.root,
                from_=-5.0,
                to=5.0,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                command=lambda v, i=len(sliders): self.update_slice(i),
            )
            slider.pack()
            sliders.append(slider)

        return sliders

    def _get_current_slice(self) -> np.ndarray:
        """Get current slice based on slider positions."""
        pos = self.sliders[self.active_slider].get()
        slice_img = slice_3d_volume(
            self.atlas, pos, self.x_angle, self.y_angle, self.active_slider
        )
        return cv2.rotate(slice_img, cv2.ROTATE_90_CLOCKWISE)

    def update_slice(self, slider_num: int) -> None:
        """Update display when sliders change.

        Args:
            slider_num: Index of changed slider.
        """
        if slider_num not in [3, 4]:
            self.active_slider = slider_num
            for i, slider in enumerate(self.sliders):
                if i != self.active_slider:
                    slider.set(0)

        self.x_angle = self.sliders[3].get()
        self.y_angle = self.sliders[4].get()

        self.ax.clear()
        self.ax.imshow(self._get_current_slice(), cmap="gray")
        self.fig.canvas.draw()

    def save_image(self) -> None:
        """Save current slice as PNG."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, self._get_current_slice())

    def run(self) -> None:
        """Start the viewer."""
        self.root.mainloop()
