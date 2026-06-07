"""triple patch GGUF を 永 続 ファイル と し て 焼 き 込 む。

入 力: google_gemma-4-31B-it-Q2_K.gguf ((原 本))
出 力: publish/release_v1/gemma-4-31B-it-tripleLOS27-PAN17-PAN43-x1.8-Q2_K.gguf

triple cells:
- layer_output_scale L27 ×1.8
- post_attention_norm L17 ×1.8
- post_attention_norm L43 ×1.8
"""
import os, sys, shutil, struct, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight_analysis.exp157_31b_q2_7knob_stable_odin import (
    find_all_f32_tensors, read_orig_vector, write_vector,
    BASE_GGUF,
)
import numpy as np

OUT_GGUF = "publish/release_v1/gemma-4-31B-it-tripleLOS27-PAN17-PAN43-x1.8-Q2_K.gguf"

TRIPLE = [
    ("layer_output_scale", 27, 1.8),
    ("post_attention_norm", 17, 1.8),
    ("post_attention_norm", 43, 1.8),
]


def main():
    print(f"=== bake triple patch ===", flush=True)
    print(f"  base : {BASE_GGUF}", flush=True)
    print(f"  out  : {OUT_GGUF}", flush=True)
    if not os.path.exists(BASE_GGUF):
        print(f"[error] base not found", flush=True); sys.exit(1)

    out_dir = os.path.dirname(OUT_GGUF)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(OUT_GGUF):
        sz = os.path.getsize(OUT_GGUF)
        print(f"[warn] output already exists ({sz/1e9:.2f} GB), overwriting", flush=True)

    print(f"\n[copy] base -> out ({os.path.getsize(BASE_GGUF)/1e9:.2f} GB) ...", flush=True)
    t0 = time.time()
    shutil.copy(BASE_GGUF, OUT_GGUF)
    print(f"  copy done ({time.time()-t0:.0f}s)", flush=True)

    # Inventory F32 tensors using the OUTPUT file (apply in place)
    print(f"\n[inventory] scanning F32 tensors in output...", flush=True)
    # find_all_f32_tensors uses BASE_GGUF internally, but offsets are identical
    inv = find_all_f32_tensors()
    print(f"  found {len(inv)} F32 tensors", flush=True)

    print(f"\n[apply] triple patch:", flush=True)
    for cat, layer, K in TRIPLE:
        if (cat, layer) not in inv:
            print(f"  [error] {cat}/L{layer} not in inventory", flush=True); sys.exit(1)
        info = inv[(cat, layer)]
        # Read from BASE (since SHARED_GGUF is BASE_GGUF, we read original)
        orig = read_orig_vector(info["offset"], info["n_elements"])
        new = orig * K
        # Write to OUT_GGUF (write_vector writes to SHARED_GGUF by default - need to redirect)
        # Use struct + raw file write at the right offset
        with open(OUT_GGUF, "rb+") as f:
            f.seek(info["offset"])
            data = struct.pack(f"<{len(new)}f", *new.astype(np.float32))
            f.write(data)
        print(f"  {cat}/L{layer} ×{K}: mean {orig.mean():.4f} -> {new.mean():.4f} ({info['n_elements']} elements)", flush=True)

    # Verify by re-reading
    print(f"\n[verify] re-reading patched values...", flush=True)
    for cat, layer, K in TRIPLE:
        info = inv[(cat, layer)]
        with open(OUT_GGUF, "rb") as f:
            f.seek(info["offset"])
            data = f.read(info["n_elements"] * 4)
            check = np.frombuffer(data, dtype=np.float32)
        orig = read_orig_vector(info["offset"], info["n_elements"])
        expected = orig * K
        diff = np.abs(check - expected).max()
        ok = "★OK" if diff < 1e-6 else "FAIL"
        print(f"  {cat}/L{layer}: max diff = {diff:.6e} [{ok}]", flush=True)

    print(f"\n[done] {OUT_GGUF}", flush=True)
    print(f"  size: {os.path.getsize(OUT_GGUF)/1e9:.2f} GB", flush=True)
    print(f"  patch: 12 byte total ((3 cells × 4 byte))", flush=True)
    print(f"\n使 い 方:", flush=True)
    print(f"  llama.cpp / LM Studio / koboldcpp 等 で 直 接 load 可", flush=True)
    print(f"  例: llama-cli -m {OUT_GGUF} -p 'hello' -ngl 99", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
