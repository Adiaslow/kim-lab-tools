"""Image conversion utilities."""

import numpy as np
from qtpy.QtGui import QImage


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Convert a numpy array to a QImage.

    Args:
        array: Input numpy array.

    Returns:
        QImage: Converted image.

    Raises:
        ValueError: If array shape is unsupported.
    """
    if np.ndim(array) == 3:
        h, w, ch = array.shape
        if array.flags["C_CONTIGUOUS"]:
            array = array.copy(order="C")
        if ch == 3:
            format = QImage.Format.Format_RGB888
        elif ch == 4:
            format = QImage.Format.Format_ARGB32
        else:
            raise ValueError(f"Unsupported channel number: {ch}")
    elif np.ndim(array) == 2:
        h, w = array.shape
        format = QImage.Format.Format_Grayscale8
    else:
        raise ValueError(f"Unsupported numpy array shape: {array.shape}")

    qimage = QImage(array.data, w, h, array.strides[0], format)
    qimage.ndarray = array  # Keep reference to prevent garbage collection
    return qimage


def qimage_to_numpy(qimage: QImage) -> np.ndarray:
    """Convert a QImage to a numpy array.

    Args:
        qimage: Input QImage.

    Returns:
        np.ndarray: Converted array.
    """
    qimage = qimage.convertToFormat(QImage.Format.Format_RGB32)
    width = qimage.width()
    height = qimage.height()

    ptr = qimage.bits()
    ptr.setsize(height * width * 4)
    return np.array(ptr).reshape((height, width, 4))
