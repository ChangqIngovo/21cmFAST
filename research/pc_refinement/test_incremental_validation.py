import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from incremental_validation import run


def test_same_shape_reference_with_unknown_origin_is_rejected(tmp_path):
    s = np.ones((2,)*3)
    fields = tmp_path/"fields.npz"
    ref = tmp_path/"ref.npz"
    np.savez(fields, source=s, capacity=2*s, initial=0*s)
    np.savez(ref, reference=s)
    with pytest.raises(ValueError, match="provenance"):
        run(fields, ref, ref, tmp_path/"result.json")


def test_verified_input_hashes_do_not_excuse_wrong_horizon(tmp_path):
    s = np.ones((2,)*3)
    data = dict(source=s, capacity=2*s, initial=0*s)
    hashes = {key+"_sha256": hashlib.sha256(value.tobytes()).hexdigest()
              for key, value in data.items()}
    fields = tmp_path/"fields.npz"
    ref = tmp_path/"ref.npz"
    np.savez(fields, **data)
    np.savez(ref, reference=s, angles=3072, maximum_distance=2., **hashes)
    with pytest.raises(ValueError, match="horizon"):
        run(fields, ref, ref, tmp_path/"result.json")
