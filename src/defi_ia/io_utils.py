"""Crash-safe artifact writing.

A GPU fine-tune runs for hours before it reaches the point where it saves its
logits, so the save path must not be the weak link: a process killed mid-write
should leave the previous run's artifact intact rather than a truncated file.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def atomic_save(path: str | Path, write_fn: Callable[[Path], object]) -> Path:
    """Write via a temp file in the same directory, then rename into place.

    ``write_fn`` receives a temp path carrying the **same suffix** as ``path``.
    That detail is load-bearing: ``np.save`` silently appends ``.npy`` to any
    filename lacking it, so a plain ``x.npy.tmp`` temp name makes numpy write
    ``x.npy.tmp.npy`` while the rename looks for ``x.npy.tmp`` and fails. Keeping
    the suffix on the temp file avoids that whole class of surprise for numpy,
    pandas and anything else that infers a format from the extension.

    ``os.replace`` is atomic within a filesystem, so a reader either sees the old
    file or the new one — never a half-written one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
