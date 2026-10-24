from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from optimize_core import solve


@pytest.mark.parametrize("variant",["heap2-wide","heap4-wide","heap8-wide",
                                    "heap2-compact","heap4-compact","heap8-compact",
                                    "heap4-wide-aligned","heap4-compact-aligned",
                                    "calendar1024-compact","calendar4096-compact","calendar16384-compact",
                                    "local-dda-compact","local-dda-wide"])
def test_heap_and_compact_preserve_map_source_and_sparse_state(variant):
    rng=np.random.default_rng(295)
    c=np.exp(rng.normal(0,.8,size=(12,)*3))
    s=np.zeros_like(c);s[4,6,6],s[8,5,7]=150,90
    old=.1*c
    base=solve(s,c,old,variant="original",maximum_distance=5,return_dense=True)
    result=solve(s,c,old,variant=variant,maximum_distance=5,return_dense=True)
    for name in ("absorbed","source_absorbed","source_escaped","source_retained","subcell_absorbed"):
        np.testing.assert_array_equal(result[name],base[name])
    for name in ("ionized","parent_ids","child_ionized"):
        np.testing.assert_array_equal(result["state"][name],base["state"][name])
    expected=.75 if "-compact" in variant else 1.
    assert result["audit"]["ray_bytes"]==expected*base["audit"]["ray_bytes"]


def test_large_angular_bank_uses_wide_fallback():
    c=np.ones((4,)*3);s=np.zeros_like(c);s[2,2,2]=2
    options=dict(min_angles=98304,max_angles=98304,maximum_distance=1.)
    wide=solve(s,c,variant="heap4-wide",**options)
    compact=solve(s,c,variant="heap4-compact",**options)
    np.testing.assert_array_equal(wide["absorbed"],compact["absorbed"])
    assert wide["audit"]["ray_bytes"]==compact["audit"]["ray_bytes"]


def test_large_finite_support_uses_wide_fallback():
    c=np.ones((4,)*3);s=np.zeros_like(c);s[2,2,2]=2
    options=dict(min_angles=24,max_angles=24,maximum_distance=20000,periodic=False)
    wide=solve(s,c,variant="heap4-wide",**options)
    compact=solve(s,c,variant="heap4-compact",**options)
    np.testing.assert_array_equal(wide["absorbed"],compact["absorbed"])
    assert wide["audit"]["ray_bytes"]==compact["audit"]["ray_bytes"]


@pytest.mark.parametrize("buckets",[1024,4096,16384])
def test_calendar_priority_oracle(tmp_path,buckets):
    import subprocess
    source=Path(__file__).with_name("test_calendar_queue.cpp")
    binary=tmp_path/"queue_test"
    subprocess.run(["g++","-std=c++17","-O2",f"-DPC_CALENDAR_BUCKETS={buckets}",
                    str(source),"-o",str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    assert "priority oracle passed" in result.stdout
