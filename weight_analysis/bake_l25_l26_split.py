"""L25 単 体 ×1.5、 L26 単 体 ×1.5 の GGUF を 個 別 に 焼 き 込 む。

役 割 asymmetry 仮 説 ((L25 = 計 算、 L26 = メタ 思 考)) 検 証 用。

出 力:
- publish/release_v1/gemma-4-31B-it-L25x1.5-Q2_K.gguf  ((L25 単 体))
- publish/release_v1/gemma-4-31B-it-L26x1.5-Q2_K.gguf  ((L26 単 体))
"""
import os, sys, shutil, struct, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight_analysis.exp157_31b_q2_7knob_stable_odin import (
    find_all_f32_tensors, read_orig_vector, BASE_GGUF,
)
import numpy as np

CONFIGS = [
    ("publish/release_v1/gemma-4-31B-it-L25x1.5-Q2_K.gguf",
     [("layer_output_scale", 25, 1.5)]),
    ("publish/release_v1/gemma-4-31B-it-L26x1.5-Q2_K.gguf",
     [("layer_output_scale", 26, 1.5)]),
]


def bake_one(out_path, cells):
    print(f"\n=== {os.path.basename(out_path)} ===", flush=True)
    print(f"  cells: {cells}", flush=True)
    if not os.path.exists(BASE_GGUF):
        print(f"[error] base not found", flush=True); sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"  [copy] base -> out ({os.path.getsize(BASE_GGUF)/1e9:.2f} GB)...", flush=True)
    t0 = time.time()
    shutil.copy(BASE_GGUF, out_path)
    print(f"  copy done ({time.time()-t0:.0f}s)", flush=True)

    inv = find_all_f32_tensors()

    for cat, layer, K in cells:
        if (cat, layer) not in inv:
            print(f"  [error] {cat}/L{layer} not in inventory", flush=True); sys.exit(1)
        info = inv[(cat, layer)]
        orig = read_orig_vector(info["offset"], info["n_elements"])
        new = orig * K
        with open(out_path, "rb+") as f:
            f.seek(info["offset"])
            data = struct.pack(f"<{len(new)}f", *new.astype(np.float32))
            f.write(data)
        print(f"  {cat}/L{layer} ×{K}: mean {orig.mean():.4f} -> {new.mean():.4f}", flush=True)

    # Verify
    for cat, layer, K in cells:
        info = inv[(cat, layer)]
        orig = read_orig_vector(info["offset"], info["n_elements"])
        expected = orig * K
        with open(out_path, "rb") as f:
            f.seek(info["offset"])
            data = f.read(info["n_elements"] * 4)
            check = np.frombuffer(data, dtype=np.float32)
        diff = np.abs(check - expected).max()
        ok = "★OK" if diff < 1e-6 else "FAIL"
        print(f"  verify {cat}/L{layer}: max diff = {diff:.6e} [{ok}]", flush=True)

    print(f"  [done] {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)", flush=True)


def main():
    print(f"=== L25/L26 split patch bake ===", flush=True)
    for out_path, cells in CONFIGS:
        bake_one(out_path, cells)
    print(f"\n[all done]", flush=True)
    print(f"\n使 い 方:", flush=True)
    for out_path, cells in CONFIGS:
        print(f"  llama-server -m {out_path} -ngl 99 -c 32768 ...", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
