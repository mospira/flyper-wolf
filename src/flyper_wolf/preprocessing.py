"""NumPy implementation of FlyVis BoxEye mean sampling, without neural inference.

Reference: TuragaLab/flyvis, flyvis/datasets/rendering/eye.py, BoxEye.
Our image is large enough to avoid upstream's resize and zero-padding branches.
Ordering is u ascending, then v ascending. Rows increase downward.
"""

import numpy as np


class VisualInput:
    version = "boxeye-mean-v1"

    def __init__(self, image_size=(384, 384), extent=15, kernel=11):
        if len(image_size) != 2 or any(type(x) is not int or x < 1 for x in image_size):
            raise ValueError("Image size must be positive integer width and height")
        width, height = image_size
        if extent != 15 or type(kernel) is not int or kernel < 1 or kernel % 2 != 1:
            raise ValueError("Expected extent 15 and a positive odd pooling kernel")
        if min(width, height) < (2 * extent + 1) * kernel:
            raise ValueError("Image too small for sampling footprint")
        self.image_size = tuple(image_size)
        self.kernel = kernel
        grid = np.array([(u, v) for u in range(-extent, extent + 1)
                         for v in range(-extent, extent + 1) if abs(u + v) <= extent], dtype=np.int32)
        self.axial_coordinates = grid
        u, v = grid.T
        # Truncation toward zero matches torch's float-to-long conversion in BoxEye.
        rows = np.trunc(kernel * (u + v / 2)).astype(int) + height // 2
        columns = kernel * v + width // 2
        self.centers_rc = np.column_stack((rows, columns))
        r = kernel // 2
        self.top, self.left = rows - r, columns - r
        self.bottom, self.right = rows + r + 1, columns + r + 1

    def grayscale(self, rgb: np.ndarray) -> np.ndarray:
        """Validate one RGB observation and convert it to fixed [0, 1] brightness.

        Float64 working precision retains the original adapter's exact results.
        Only the final receptor values are cast to the circuit's float32 dtype.
        """
        width, height = self.image_size
        if not isinstance(rgb, np.ndarray) or rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected configured H x W x 3 uint8 RGB image")
        return rgb.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722]) / 255

    def sample_grayscale(self, gray: np.ndarray) -> np.ndarray:
        """Average each fixed square footprint; return 721 ordered float32 values."""
        width, height = self.image_size
        if (not isinstance(gray, np.ndarray) or gray.shape != (height, width)
                or not np.issubdtype(gray.dtype, np.floating)
                or not np.isfinite(gray).all() or np.any((gray < 0) | (gray > 1))):
            raise ValueError("Expected finite H x W floating-point grayscale in [0, 1]")
        # A summed-area table gives each square's mean without filtering every
        # output pixel. This is mathematically the same local box average.
        gray = gray.astype(np.float64, copy=False)
        integral = np.pad(gray.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        sums = (integral[self.bottom, self.right] - integral[self.top, self.right]
                - integral[self.bottom, self.left] + integral[self.top, self.left])
        return np.clip(sums / self.kernel**2, 0, 1).astype(np.float32)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """RGB HWC / THWC / BTHWC -> contiguous float32 [batch, time, 1, 721].

        Frames remain chronological and independent. No temporal averaging,
        contrast stretching, cropping/resizing, or hidden scene information.
        """
        width, height = self.image_size
        if (not isinstance(rgb, np.ndarray) or rgb.ndim not in (3, 4, 5)
                or rgb.shape[-3:] != (height, width, 3) or rgb.dtype != np.uint8
                or any(size == 0 for size in rgb.shape)):
            raise ValueError("Expected nonempty HWC, THWC, or BTHWC uint8 RGB at the configured image size")
        if rgb.ndim == 3:
            frames = rgb[None, None]
        elif rgb.ndim == 4:
            frames = rgb[None]
        else:
            frames = rgb
        output = np.empty((*frames.shape[:2], 1, 721), dtype=np.float32)
        # Process one full-resolution frame at a time to bound temporary memory.
        for batch in range(frames.shape[0]):
            for tick in range(frames.shape[1]):
                output[batch, tick, 0] = self.sample_grayscale(self.grayscale(frames[batch, tick]))
        return output

    def to_tensor(self, rgb: np.ndarray, device="cpu"):
        """Build an inference-only Torch tensor in FlyVis's video-input layout.

        This does not advance or reset neural state. The stateful circuit
        adapter owns that responsibility and supplies the neural integration dt.
        """
        return as_tensor(self(rgb), device=device)


def as_tensor(values: np.ndarray, device="cpu"):
    """Validate preprocessed values before the explicit circuit handoff."""
    if (not isinstance(values, np.ndarray) or values.ndim != 4
            or values.shape[2:] != (1, 721) or min(values.shape[:2]) < 1
            or values.dtype != np.float32 or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError("Expected finite float32 [batch, time, 1, 721] values in [0, 1]")
    try:
        import torch
    except ImportError as error:
        raise ImportError("Tensor handoff requires: install the runtime dependencies documented in README.md") from error
    # Own the storage: later mutations to an archived NumPy array cannot alter input.
    return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=False).contiguous()
