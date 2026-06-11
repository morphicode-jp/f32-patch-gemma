#!/usr/bin/env python3
"""apply_tsubo4.py - TSUBO-4: 4-byte F32 patch for Gemma 4 12B-it Q2_K.

Multiplies blk.10.layer_output_scale.weight by exactly x1.65.
On bartowski/gemma-4-12b-it-GGUF Q2_K the value should move:
0.104004 -> 0.171606.

Usage:
  pip install gguf numpy
  python apply_tsubo4.py /path/to/gemma-4-12B-it-Q2_K.gguf
  python apply_tsubo4.py /path/to/gemma-4-12B-it-Q2_K.gguf --restore
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader

TARGET_NAME = "blk.10.layer_output_scale.weight"
SCALE = np.float32(1.65)
BASE_VALUE = np.float32(0.1040039137005806)
PATCHED_VALUE = np.float32(BASE_VALUE * SCALE)
TOL = 2e-6


def close(a: float, b: np.float32) -> bool:
    return abs(float(a) - float(b)) <= TOL


def find_offset(gguf_path: Path) -> int:
    reader = GGUFReader(str(gguf_path))
    offset = None
    for tensor in reader.tensors:
        if str(tensor.name) == TARGET_NAME:
            if tensor.tensor_type.name != "F32" or int(tensor.n_elements) != 1:
                sys.exit(f"[error] {TARGET_NAME} is not a 1-element F32 tensor")
            offset = int(tensor.data_offset)
            break
    del reader
    if offset is None:
        sys.exit(f"[error] {TARGET_NAME} not found - is this a Gemma 4 12B GGUF?")
    return offset


def read_f32(path: Path, offset: int) -> np.float32:
    with path.open("rb") as f:
        f.seek(offset)
        return np.float32(struct.unpack("<f", f.read(4))[0])


def write_f32(path: Path, offset: int, value: np.float32) -> None:
    with path.open("r+b") as f:
        f.seek(offset)
        f.write(struct.pack("<f", float(np.float32(value))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply or restore the TSUBO-4 4-byte patch.")
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--restore", action="store_true", help="restore the original baseline value")
    parser.add_argument("--no-backup", action="store_true", help="skip creating <file>.backup before patching")
    args = parser.parse_args()

    if not args.gguf.is_file():
        sys.exit(f"[error] file not found: {args.gguf}")

    offset = find_offset(args.gguf)
    before = read_f32(args.gguf, offset)

    if args.restore:
        if close(before, BASE_VALUE):
            print(f"[TSUBO-4] already restored: {before:.9f}")
            return
        if not close(before, PATCHED_VALUE):
            sys.exit(f"[error] refusing restore: current value {before:.9f} is not TSUBO-4 patched")
        write_f32(args.gguf, offset, BASE_VALUE)
        print(f"[TSUBO-4] restored {TARGET_NAME} @ byte offset {offset}")
        print(f"[TSUBO-4] {before:.9f} -> {BASE_VALUE:.9f}")
        return

    if close(before, PATCHED_VALUE):
        print(f"[TSUBO-4] already patched: {before:.9f}")
        return
    if not close(before, BASE_VALUE):
        sys.exit(f"[error] unexpected value {before:.9f}; expected baseline {BASE_VALUE:.9f}")

    if not args.no_backup:
        backup = args.gguf.with_name(args.gguf.name + ".backup")
        if not backup.exists():
            shutil.copy2(args.gguf, backup)
            print(f"[TSUBO-4] backup created: {backup}")
    after = np.float32(before * SCALE)
    write_f32(args.gguf, offset, after)
    print(f"[TSUBO-4] {TARGET_NAME} @ byte offset {offset}")
    print(f"[TSUBO-4] before = {before:.9f} -> after = {after:.9f} (x{float(SCALE):.2f})")
    print("[TSUBO-4] 4 bytes written in place.")


if __name__ == "__main__":
    main()