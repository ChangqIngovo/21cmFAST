"""The event queue must preserve ordering during updates and arbitrary erasure."""
from pathlib import Path
import subprocess


def test_actual_native_heap_against_ordered_reference(tmp_path):
    source = Path(__file__).with_name('test_refined_heap.cpp')
    binary = tmp_path / 'heap_check'
    subprocess.run(['g++', '-O2', '-std=c++17', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
