"""apply_basin_b.py — 任意の Gemma 4 31B GGUF に basin B (44 byte F32 patch) を適用.

Usage:
  python apply_basin_b.py <path_to_gemma4_31b.gguf> [--restore]

What it does:
  - Reads F32 layer_output_scale tensors from 11 specific layers
  - Multiplies each by basin B scale (range 0.55-1.44)
  - Writes back in-place (creates .backup of original)
  - 44 bytes total modification (11 layers × 4 bytes F32)
  - --restore: undo, restore from .backup

Quant 対応: 全 Gemma 4 31B 量子化 (IQ1_M / Q2_K / Q4_K_M / Q5_K_M / Q8_0 等)
- F32 layer_output_scale は量子化に依らず全 GGUF で同じ位置に存在

Discovered via ODIN autonomous optimization engine (12 specialist parallel search).
Verified across HellaSwag (n=10042), GSM8k (n=100), Winogrande (full).
Alignment preserved on AdvBench (n=50, 100% refusal rate).

License: Apache 2.0
Author: Akito Hirai (@morphicode_jp)
Repo: https://github.com/morphicode_jp/f32-patch-gemma
"""
import argparse
import hashlib
import os
import shutil
import struct
import sys
from pathlib import Path

# basin B (ODIN autonomous discovery, 11 layers × scale)
BASIN_B_PATCH = {
    5:  1.440,
    18: 0.767,
    19: 1.016,
    20: 0.814,
    21: 1.413,
    25: 1.402,
    26: 1.289,
    27: 1.326,
    28: 1.431,
    30: 0.548,
    31: 1.267,
}


def find_layer_output_scale_offsets(gguf_path: str) -> dict:
    """GGUF を読んで blk.{layer}.layer_output_scale.weight の (offset, n_elements) 辞書を返す."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("[error] gguf package required. install: pip install gguf", file=sys.stderr)
        sys.exit(1)

    reader = GGUFReader(gguf_path)
    offsets = {}
    for t in reader.tensors:
        name = str(t.name)
        # Gemma 4: blk.{N}.layer_output_scale.weight (F32, 1 element)
        if not name.startswith("blk."):
            continue
        parts = name.split(".", 2)
        if len(parts) < 3:
            continue
        try:
            layer = int(parts[1])
        except ValueError:
            continue
        rest = parts[2]
        if rest != "layer_output_scale.weight":
            continue
        if t.tensor_type.name != "F32":
            continue
        if int(t.n_elements) != 1:
            continue
        offsets[layer] = {
            "offset": int(t.data_offset),
            "n_elements": int(t.n_elements),
        }
    del reader
    return offsets


def read_current_value(gguf_path: str, offset: int) -> float:
    with open(gguf_path, "rb") as f:
        f.seek(offset)
        return struct.unpack("<f", f.read(4))[0]


def write_value(gguf_path: str, offset: int, value: float) -> None:
    with open(gguf_path, "r+b") as f:
        f.seek(offset)
        f.write(struct.pack("<f", float(value)))


def md5sum_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            d = f.read(chunk_size)
            if not d:
                break
            h.update(d)
    return h.hexdigest()


def apply_patch(gguf_path: str, dry_run: bool = False) -> None:
    print(f"[info] Reading GGUF structure: {gguf_path}")
    print(f"[info] File size: {os.path.getsize(gguf_path) / 1e9:.2f} GB")

    offsets = find_layer_output_scale_offsets(gguf_path)
    if len(offsets) < 60:
        print(f"[warn] Found only {len(offsets)} layer_output_scale tensors. "
              f"Expected 60 (Gemma 4 31B). This may not be Gemma 4 31B.")
        if not dry_run:
            resp = input("Continue anyway? [y/N]: ").strip().lower()
            if resp != "y":
                print("[abort]")
                return

    print(f"[info] Found {len(offsets)} layer_output_scale F32 tensors")

    missing = [l for l in BASIN_B_PATCH if l not in offsets]
    if missing:
        print(f"[error] Missing target layers: {missing}", file=sys.stderr)
        sys.exit(2)

    print(f"\n[info] basin B patch plan (11 layers, 44 bytes total):")
    print(f"  {'layer':>5} | {'offset':>10} | {'current':>10} | {'new':>10}")
    print(f"  {'-'*5} | {'-'*10} | {'-'*10} | {'-'*10}")

    plan = []
    for layer, scale in sorted(BASIN_B_PATCH.items()):
        off = offsets[layer]["offset"]
        cur = read_current_value(gguf_path, off)
        new = cur * scale
        plan.append((layer, off, cur, scale, new))
        print(f"  {layer:>5} | {off:>10} | {cur:>10.4f} | {new:>10.4f}  (×{scale})")

    if dry_run:
        print(f"\n[dry-run] No changes applied.")
        return

    # Backup
    backup_path = gguf_path + ".backup"
    if not os.path.exists(backup_path):
        print(f"\n[info] Creating backup: {backup_path}")
        shutil.copy2(gguf_path, backup_path)
    else:
        print(f"\n[info] Backup already exists: {backup_path}")

    # Apply
    print(f"\n[info] Applying patch...")
    for layer, off, cur, scale, new in plan:
        write_value(gguf_path, off, new)
    print(f"[done] Applied basin B patch to {len(plan)} layers.")

    # MD5
    print(f"\n[info] Computing MD5 of patched file (may take 30s)...")
    md5 = md5sum_file(gguf_path)
    print(f"[done] MD5: {md5}")
    print(f"\n[next] Test with llama-perplexity or your usual eval pipeline.")
    print(f"[next] To restore original: python apply_basin_b.py {gguf_path} --restore")


def restore_from_backup(gguf_path: str) -> None:
    backup_path = gguf_path + ".backup"
    if not os.path.exists(backup_path):
        print(f"[error] Backup not found: {backup_path}", file=sys.stderr)
        print(f"[hint] basin B was not applied via this script, or backup was deleted.")
        sys.exit(2)
    print(f"[info] Restoring from backup: {backup_path}")
    shutil.copy2(backup_path, gguf_path)
    print(f"[done] Restored original GGUF.")


def main():
    parser = argparse.ArgumentParser(
        description="Apply basin B (44-byte F32 patch) to Gemma 4 31B GGUF.",
    )
    parser.add_argument("gguf_path", help="Path to Gemma 4 31B GGUF file")
    parser.add_argument("--restore", action="store_true",
                         help="Restore from .backup (undo previous patch)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Show plan without modifying file")
    args = parser.parse_args()

    if not os.path.exists(args.gguf_path):
        print(f"[error] File not found: {args.gguf_path}", file=sys.stderr)
        sys.exit(1)

    if args.restore:
        restore_from_backup(args.gguf_path)
    else:
        apply_patch(args.gguf_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
