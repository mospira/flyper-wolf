"""Scoped compatibility fixes for the pinned FlyVis/datamate runtime."""

from contextlib import contextmanager
from importlib.metadata import version
import sys


def _write_h5_closed(path, value):
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    # Upstream opens mode='w', reads a nonexistent dataset, then attempts to
    # unlink the still-open file. Create the dataset directly and always close.
    with h5py.File(path, mode="w", libver="latest") as handle:
        handle.create_dataset("data", data=np.asarray(value))
        handle.swmr_mode = True


@contextmanager
def windows_hdf5_writer():
    """Patch only during synchronous construction; restore even on failure.

    No installed packages or HDF5 locking settings are modified. Construction
    must not run concurrently with other datamate writes in this process.
    """
    if sys.platform != "win32" or version("datamate") != "1.0.0":
        yield False
        return
    import datamate.directory as directory
    import datamate.io as io

    original_io, original_directory = io._write_h5, directory._write_h5
    io._write_h5 = directory._write_h5 = _write_h5_closed
    try:
        yield True
    finally:
        io._write_h5, directory._write_h5 = original_io, original_directory


@contextmanager
def flyvis_initialization_device(device):
    """Match the initializer's RNG to the requested tensor device.

    FlyVis 1.2.0 captures CUDA availability globally at import. Without this,
    explicitly constructing a CPU circuit on a CUDA machine creates a CUDA
    generator for CPU tensors. Like the writer fix, construction is synchronous.
    """
    from importlib import import_module

    initialization = import_module("flyvis.network.initialization")

    previous = initialization.device
    initialization.device = device
    try:
        yield
    finally:
        initialization.device = previous
