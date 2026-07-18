"""Regression guard for the atomic-write helper used by the training scripts.

A GPU fine-tune can run for hours before it reaches the point where it saves its
logits. If the save path is broken, everything is lost at the last step. The
first version of ``_atomic_save`` passed a temp path ending in ``.tmp``, and
``np.save`` silently appends ``.npy`` to any filename that lacks that suffix —
so numpy wrote ``valid_logits.npy.tmp.npy`` while ``os.replace`` looked for
``valid_logits.npy.tmp`` and raised FileNotFoundError.

These tests pin the behaviour that fix depends on, so it cannot regress
unnoticed the next time someone touches the helper.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from defi_ia.io_utils import atomic_save as _atomic_save


def test_atomic_save_npy_lands_at_the_requested_path(tmp_path):
    """np.save must not smuggle an extra .npy onto the temp file."""
    target = tmp_path / "valid_logits.npy"
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)

    _atomic_save(target, lambda p: np.save(p, arr))

    assert target.exists(), "logits were not written to the requested path"
    np.testing.assert_array_equal(np.load(target), arr)


def test_atomic_save_leaves_no_temp_files_behind(tmp_path):
    target = tmp_path / "test_logits.npy"
    _atomic_save(target, lambda p: np.save(p, np.zeros((2, 2))))

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != target.name]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_atomic_save_handles_csv_and_json(tmp_path):
    """The helper is generic — the other two artifact types must work too."""
    csv_target = tmp_path / "valid_meta.csv"
    df = pd.DataFrame({"Category": [1, 2], "gender": ["M", "F"]}, index=[10, 11])
    _atomic_save(csv_target, lambda p: df.to_csv(p, index_label="Id"))
    assert pd.read_csv(csv_target, index_col="Id")["gender"].tolist() == ["M", "F"]

    json_target = tmp_path / "metrics.json"
    _atomic_save(json_target, lambda p: p.write_text(json.dumps({"macro_f1": 0.8})))
    assert json.loads(json_target.read_text())["macro_f1"] == 0.8


def test_atomic_save_preserves_the_previous_file_when_writing_fails(tmp_path):
    """A crash mid-write must not destroy the artifact from the last good run."""
    target = tmp_path / "valid_logits.npy"
    _atomic_save(target, lambda p: np.save(p, np.ones((2, 2))))

    def boom(_p):
        raise RuntimeError("simulated crash during save")

    try:
        _atomic_save(target, boom)
    except RuntimeError:
        pass

    np.testing.assert_array_equal(np.load(target), np.ones((2, 2)))
