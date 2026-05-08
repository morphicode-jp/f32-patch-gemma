# Zenron-EML Smoke Benchmark v8

Command:

```powershell
python experiments/zenron_eml_bridge/run_eml_zenron_smoke.py
```

## Aggregate Result Table

| target | mode | success | mean gens | mean evals | mean test loss | eval speedup vs raw budget | best expr examples |
|---|---:|---:|---:|---:|---:|---:|---|
| cos(x) | kathara_lad | 0/5 | - | - | - | -x | - |
| cos(x) | kathara_mined | 0/5 | - | - | - | -x | - |
| cos(x) | kathara_mined_no_cert | 0/5 | - | - | - | -x | - |
| cos(x) | kathara_mined_no_seed | 0/5 | - | - | - | -x | - |
| cos(x) | kathara_mined_no_weight | 0/5 | - | - | - | -x | - |
| cos(x) | lad_seeded | 0/5 | - | - | - | -x | - |
| cos(x) | macro_mined | 0/5 | - | - | - | -x | - |
| cos(x) | mined_no_cert | 0/5 | - | - | - | -x | - |
| cos(x) | mined_no_seed | 0/5 | - | - | - | -x | - |
| cos(x) | mined_no_weight | 0/5 | - | - | - | -x | - |
| cos(x) | raw_eml | 0/5 | - | - | - | -x | - |
| exp(log(x)) | kathara_lad | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| exp(log(x)) | kathara_mined | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| exp(log(x)) | kathara_mined_no_cert | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| exp(log(x)) | kathara_mined_no_seed | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| exp(log(x)) | kathara_mined_no_weight | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| exp(log(x)) | lad_seeded | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| exp(log(x)) | macro_mined | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| exp(log(x)) | mined_no_cert | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| exp(log(x)) | mined_no_seed | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| exp(log(x)) | mined_no_weight | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| exp(log(x)) | raw_eml | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| exp(x) | kathara_lad | 5/5 | 1.0 | 144.0 | 3.00e-10 | 0.25x | `E(x, 1)` |
| exp(x) | kathara_mined | 5/5 | 1.0 | 144.0 | 3.00e-10 | 0.25x | `E(x, 1)` |
| exp(x) | kathara_mined_no_cert | 5/5 | 1.0 | 144.0 | 3.00e-10 | 0.25x | `E(x, 1)` |
| exp(x) | kathara_mined_no_seed | 5/5 | 1.0 | 144.0 | 3.00e-10 | 0.25x | `E(x, 1)` |
| exp(x) | kathara_mined_no_weight | 5/5 | 1.0 | 144.0 | 3.00e-10 | 0.25x | `E(x, 1)` |
| exp(x) | lad_seeded | 5/5 | 1.0 | 24.0 | 3.00e-10 | 1.50x | `E(x, 1)` |
| exp(x) | macro_mined | 5/5 | 9.8 | 129.6 | 1.02e-09 | 1.14x | `E(E(1, E(E(1, 1), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, E(1, x)), 1)), E(E(x, x), 1))), 1)), E(E(1, E(1, 1)), 1)), 1))`<br>`E(x, 1)` |
| exp(x) | mined_no_cert | 5/5 | 2.0 | 36.0 | 3.00e-10 | 1.13x | `E(x, 1)` |
| exp(x) | mined_no_seed | 5/5 | 1.8 | 33.6 | 3.00e-10 | 1.13x | `E(x, 1)` |
| exp(x) | mined_no_weight | 5/5 | 6.4 | 88.8 | 1.62e-09 | 0.74x | `E(E(1, E(E(1, E(E(1, E(E(1, x), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(x, 1))), 1)), E(1, 1)), 1))), E(1, E(E(1, 1), 1)))), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(x, 1)` |
| exp(x) | raw_eml | 5/5 | 2.0 | 36.0 | 3.00e-10 | 1.00x | `E(x, 1)` |
| exp(x)+log(x) | kathara_lad | 5/5 | 1.6 | 223.2 | 3.90e-09 | 15.56x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | kathara_mined | 5/5 | 4.4 | 592.8 | 3.90e-09 | 8.38x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | kathara_mined_no_cert | 5/5 | 1.8 | 249.6 | 4.06e-09 | 13.64x | `E(E(1, E(E(1, E(1, E(E(1, x), E(E(1, E(E(1, 1), 1)), 1)))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | kathara_mined_no_seed | 5/5 | 3.4 | 460.8 | 3.90e-09 | 9.11x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | kathara_mined_no_weight | 5/5 | 10.2 | 1358.4 | 4.50e-09 | 3.97x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(E(1, E(E(1, E(x, E(E(x, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, E(1, E(E(1, x), 1))), 1))), 1)), E(1, 1)), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, E(E(x, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(1, E(E(1, 1), 1))), 1)), x), 1))` |
| exp(x)+log(x) | lad_seeded | 4/5 | 10.8 | 141.0 | 3.90e-09 | 51.93x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | macro_mined | 5/5 | 20.2 | 254.4 | 4.22e-09 | 12.75x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(E(E(1, E(E(1, x), 1)), 1), 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | mined_no_cert | 5/5 | 16.6 | 211.2 | 3.90e-09 | 20.90x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | mined_no_seed | 5/5 | 10.2 | 134.4 | 3.90e-09 | 36.55x | `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | mined_no_weight | 4/5 | 47.0 | 576.0 | 4.30e-09 | 8.99x | `E(E(1, E(E(1, E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(E(1, E(E(1, E(x, 1)), 1)), 1), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))` |
| exp(x)+log(x) | raw_eml | 0/5 | - | - | - | -x | - |
| log(exp(x)) | kathara_lad | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| log(exp(x)) | kathara_mined | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| log(exp(x)) | kathara_mined_no_cert | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| log(exp(x)) | kathara_mined_no_seed | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| log(exp(x)) | kathara_mined_no_weight | 5/5 | 1.0 | 144.0 | 1.00e-10 | 0.17x | `x` |
| log(exp(x)) | lad_seeded | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| log(exp(x)) | macro_mined | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| log(exp(x)) | mined_no_cert | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| log(exp(x)) | mined_no_seed | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| log(exp(x)) | mined_no_weight | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| log(exp(x)) | raw_eml | 5/5 | 1.0 | 24.0 | 1.00e-10 | 1.00x | `x` |
| log(log(x)) | kathara_lad | 4/5 | 2.0 | 276.0 | 1.50e-09 | 8.02x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))`<br>`E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), 1)` |
| log(log(x)) | kathara_mined | 5/5 | 2.6 | 355.2 | 1.30e-09 | 8.12x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))`<br>`E(x, E(E(x, E(1, E(E(1, x), 1))), 1))` |
| log(log(x)) | kathara_mined_no_cert | 4/5 | 2.2 | 309.0 | 1.30e-09 | 7.31x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` |
| log(log(x)) | kathara_mined_no_seed | 4/5 | 2.2 | 309.0 | 1.30e-09 | 6.83x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` |
| log(log(x)) | kathara_mined_no_weight | 4/5 | 1.8 | 243.0 | 1.30e-09 | 10.03x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))`<br>`E(x, E(E(x, E(1, E(E(1, x), 1))), 1))` |
| log(log(x)) | lad_seeded | 4/5 | 6.2 | 87.0 | 1.30e-09 | 34.54x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` |
| log(log(x)) | macro_mined | 4/5 | 10.0 | 132.0 | 1.50e-09 | 27.02x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))`<br>`E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), 1)` |
| log(log(x)) | mined_no_cert | 4/5 | 33.2 | 411.0 | 1.30e-09 | 17.08x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` |
| log(log(x)) | mined_no_seed | 5/5 | 46.2 | 566.4 | 2.14e-09 | 12.15x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))`<br>`E(E(E(1, E(E(E(1, 1), E(1, E(E(1, E(1, E(x, 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(1, E(E(1, E(1, E(E(1, x), 1))), 1)))), 1)), E(1, 1)), 1)), 1)` |
| log(log(x)) | mined_no_weight | 4/5 | 4.5 | 66.0 | 1.30e-09 | 45.60x | `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` |
| log(log(x)) | raw_eml | 2/5 | 111.0 | 1344.0 | 1.30e-09 | 1.00x | `E(1, E(E(1, E(x, E(E(x, x), 1))), 1))`<br>`E(x, E(E(x, E(1, E(E(1, x), 1))), 1))` |
| log(x) | kathara_lad | 5/5 | 1.0 | 144.0 | 7.00e-10 | 6.22x | `E(1, E(E(1, x), 1))` |
| log(x) | kathara_mined | 5/5 | 1.0 | 144.0 | 7.00e-10 | 6.22x | `E(1, E(E(1, x), 1))` |
| log(x) | kathara_mined_no_cert | 5/5 | 6.6 | 883.2 | 8.60e-10 | 3.17x | `E(1, E(E(1, x), 1))`<br>`E(1, E(E(1, x), E(E(1, E(E(1, 1), 1)), 1)))` |
| log(x) | kathara_mined_no_seed | 5/5 | 1.6 | 223.2 | 7.00e-10 | 4.14x | `E(1, E(E(1, x), 1))` |
| log(x) | kathara_mined_no_weight | 5/5 | 4.2 | 566.4 | 7.00e-10 | 2.14x | `E(1, E(E(1, x), 1))`<br>`E(x, E(E(x, x), 1))` |
| log(x) | lad_seeded | 5/5 | 1.0 | 24.0 | 7.00e-10 | 37.30x | `E(1, E(E(1, x), 1))` |
| log(x) | macro_mined | 5/5 | 21.2 | 266.4 | 1.58e-09 | 13.45x | `E(1, E(E(1, x), 1))`<br>`E(E(1, E(E(1, E(E(1, E(E(1, E(1, E(E(1, 1), 1))), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| log(x) | mined_no_cert | 5/5 | 4.6 | 67.2 | 7.00e-10 | 25.46x | `E(1, E(E(1, x), 1))` |
| log(x) | mined_no_seed | 5/5 | 2.2 | 38.4 | 7.00e-10 | 24.75x | `E(1, E(E(1, x), 1))` |
| log(x) | mined_no_weight | 5/5 | 3.0 | 48.0 | 7.00e-10 | 20.81x | `E(1, E(E(1, x), 1))` |
| log(x) | raw_eml | 5/5 | 73.6 | 895.2 | 7.00e-10 | 1.00x | `E(1, E(E(1, x), 1))`<br>`E(x, E(E(x, x), 1))` |
| sin(x) | kathara_lad | 0/5 | - | - | - | -x | - |
| sin(x) | kathara_mined | 0/5 | - | - | - | -x | - |
| sin(x) | kathara_mined_no_cert | 0/5 | - | - | - | -x | - |
| sin(x) | kathara_mined_no_seed | 0/5 | - | - | - | -x | - |
| sin(x) | kathara_mined_no_weight | 0/5 | - | - | - | -x | - |
| sin(x) | lad_seeded | 0/5 | - | - | - | -x | - |
| sin(x) | macro_mined | 0/5 | - | - | - | -x | - |
| sin(x) | mined_no_cert | 0/5 | - | - | - | -x | - |
| sin(x) | mined_no_seed | 0/5 | - | - | - | -x | - |
| sin(x) | mined_no_weight | 0/5 | - | - | - | -x | - |
| sin(x) | raw_eml | 0/5 | - | - | - | -x | - |
| tanh(x) | kathara_lad | 0/5 | - | - | - | -x | - |
| tanh(x) | kathara_mined | 0/5 | - | - | - | -x | - |
| tanh(x) | kathara_mined_no_cert | 0/5 | - | - | - | -x | - |
| tanh(x) | kathara_mined_no_seed | 0/5 | - | - | - | -x | - |
| tanh(x) | kathara_mined_no_weight | 0/5 | - | - | - | -x | - |
| tanh(x) | lad_seeded | 0/5 | - | - | - | -x | - |
| tanh(x) | macro_mined | 0/5 | - | - | - | -x | - |
| tanh(x) | mined_no_cert | 0/5 | - | - | - | -x | - |
| tanh(x) | mined_no_seed | 0/5 | - | - | - | -x | - |
| tanh(x) | mined_no_weight | 0/5 | - | - | - | -x | - |
| tanh(x) | raw_eml | 0/5 | - | - | - | -x | - |
| x*log(x) | kathara_lad | 5/5 | 2.2 | 302.4 | 5.10e-09 | 11.04x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | kathara_mined | 5/5 | 2.4 | 328.8 | 5.10e-09 | 9.12x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | kathara_mined_no_cert | 5/5 | 31.2 | 4130.4 | 5.26e-09 | 4.76x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, E(x, 1)), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | kathara_mined_no_seed | 5/5 | 10.6 | 1411.2 | 5.10e-09 | 13.56x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | kathara_mined_no_weight | 4/5 | 14.8 | 1959.0 | 5.30e-09 | 8.04x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | lad_seeded | 5/5 | 12.4 | 160.8 | 5.58e-09 | 53.65x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, E(x, 1)), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | macro_mined | 5/5 | 33.8 | 417.6 | 5.26e-09 | 24.65x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | mined_no_cert | 5/5 | 21.4 | 268.8 | 5.10e-09 | 16.57x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | mined_no_seed | 5/5 | 14.8 | 189.6 | 5.26e-09 | 16.28x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | mined_no_weight | 5/5 | 26.4 | 328.8 | 5.58e-09 | 12.21x | `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x*log(x) | raw_eml | 0/5 | - | - | - | -x | - |
| x^2 | kathara_lad | 5/5 | 1.0 | 144.0 | 4.50e-09 | 20.08x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | kathara_mined | 5/5 | 1.6 | 223.2 | 4.50e-09 | 14.32x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | kathara_mined_no_cert | 5/5 | 2.0 | 276.0 | 4.50e-09 | 10.48x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | kathara_mined_no_seed | 5/5 | 1.0 | 144.0 | 4.50e-09 | 20.08x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | kathara_mined_no_weight | 5/5 | 1.8 | 249.6 | 4.50e-09 | 13.64x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | lad_seeded | 5/5 | 1.0 | 24.0 | 4.50e-09 | 120.50x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | macro_mined | 5/5 | 8.2 | 110.4 | 4.50e-09 | 30.86x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | mined_no_cert | 5/5 | 15.0 | 192.0 | 4.50e-09 | 38.98x | `E(E(E(1, E(E(1, 1), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(E(1, E(E(1, x), 1)), 1)), 1)), 1)`<br>`E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | mined_no_seed | 5/5 | 11.0 | 144.0 | 4.50e-09 | 60.82x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | mined_no_weight | 5/5 | 5.6 | 79.2 | 4.50e-09 | 56.84x | `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` |
| x^2 | raw_eml | 0/5 | - | - | - | -x | - |
| x^2+log(x) | kathara_lad | 4/5 | 6.2 | 837.0 | 8.10e-09 | 6.90x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| x^2+log(x) | kathara_mined | 5/5 | 2.8 | 381.6 | 8.10e-09 | 9.67x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| x^2+log(x) | kathara_mined_no_cert | 2/5 | 4.5 | 606.0 | 8.10e-09 | 4.83x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| x^2+log(x) | kathara_mined_no_seed | 5/5 | 17.2 | 2282.4 | 8.10e-09 | 7.08x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| x^2+log(x) | kathara_mined_no_weight | 4/5 | 16.0 | 2124.0 | 8.10e-09 | 2.76x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| x^2+log(x) | lad_seeded | 4/5 | 64.2 | 783.0 | 7.90e-09 | 22.41x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(1, E(E(1, E(x, 1)), 1)))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), x)), 1)), E(1, 1)), 1))` |
| x^2+log(x) | macro_mined | 5/5 | 31.4 | 388.8 | 8.10e-09 | 9.39x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` |
| x^2+log(x) | mined_no_cert | 0/5 | - | - | - | -x | - |
| x^2+log(x) | mined_no_seed | 5/5 | 30.8 | 381.6 | 8.26e-09 | 15.58x | `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))`<br>`E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1), 1))), 1)), E(1, 1)), 1))` |
| x^2+log(x) | mined_no_weight | 0/5 | - | - | - | -x | - |
| x^2+log(x) | raw_eml | 0/5 | - | - | - | -x | - |

## Ablation Focus Table

This table isolates which Zenron-EML accelerators matter on hard targets.

| target | macro_mined | no weight | no seed | no certificate | kathara_mined | kathara no weight | kathara no seed | kathara no certificate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `log(log(x))` | 4/5, 10.0 gen / 132.0 evals | 4/5, 4.5 gen / 66.0 evals | 5/5, 46.2 gen / 566.4 evals | 4/5, 33.2 gen / 411.0 evals | 5/5, 2.6 gen / 355.2 evals | 4/5, 1.8 gen / 243.0 evals | 4/5, 2.2 gen / 309.0 evals | 4/5, 2.2 gen / 309.0 evals |
| `x*log(x)` | 5/5, 33.8 gen / 417.6 evals | 5/5, 26.4 gen / 328.8 evals | 5/5, 14.8 gen / 189.6 evals | 5/5, 21.4 gen / 268.8 evals | 5/5, 2.4 gen / 328.8 evals | 4/5, 14.8 gen / 1959.0 evals | 5/5, 10.6 gen / 1411.2 evals | 5/5, 31.2 gen / 4130.4 evals |
| `exp(x)+log(x)` | 5/5, 20.2 gen / 254.4 evals | 4/5, 47.0 gen / 576.0 evals | 5/5, 10.2 gen / 134.4 evals | 5/5, 16.6 gen / 211.2 evals | 5/5, 4.4 gen / 592.8 evals | 5/5, 10.2 gen / 1358.4 evals | 5/5, 3.4 gen / 460.8 evals | 5/5, 1.8 gen / 249.6 evals |
| `x^2+log(x)` | 5/5, 31.4 gen / 388.8 evals | 0/5 | 5/5, 30.8 gen / 381.6 evals | 0/5 | 5/5, 2.8 gen / 381.6 evals | 4/5, 16.0 gen / 2124.0 evals | 5/5, 17.2 gen / 2282.4 evals | 2/5, 4.5 gen / 606.0 evals |

## Reading

- `raw_eml` uses only terminals `{1, x}` and `E(a,b)` mutations.
- `lad_seeded` keeps the same pure EML representation, but starts with hand witness macros for `exp` and `log`.
- `kathara_lad` adds the Kathara 12-node / 30-edge neighbor sharing loop on top of hand witness macros.
- `macro_mined` starts without hand `exp/log` witnesses and grows a cross-target macro bank from successful expressions.
- `kathara_mined` combines the mined macro bank with Kathara sharing.
- v5 gives mined macros priority weights: repeated full successes and reusable subtrees are sampled more often.
- v6 also seeds the starting population from LaD memory, so remembered subtrees begin on the Kathara workbench.
- v7 distills compact EML certificates after a mined mode succeeds, turning deep discoveries into reusable smaller macros.
- v8 adds ablation modes: `*_no_weight`, `*_no_seed`, and `*_no_cert`.
- Aggregate rows are over fixed deterministic seeds.
- If `raw_eml` misses in a trial, speedup is computed against the raw budget spent before the miss.
- Unsupported rows are explicit misses, not hidden failures.

## Mined Macro Banks

### trial 0 / macro_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 13 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, 1), 1)` | 2 | 5 | 10 | part | exp(x)+log(x) |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 12 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))` | 4 | 13 | 8 | part | exp(x)+log(x) |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1)))` | 5 | 15 | 7 | part | exp(x)+log(x) |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 20 | full, certificate | log(log(x)), log(log(x)):compact_certificate |
| `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))` | 10 | 39 | 15 | full | exp(x)+log(x) |
| `E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 39 | 19 | certificate | exp(x)+log(x):compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 45 | 20 | full, certificate | x^2, x^2:compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 51 | 15 | full | x*log(x) |
| ... | ... | ... | ... | ... | 2 more macros in JSON |

### trial 0 / kathara_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 13 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, 1), 1)` | 2 | 5 | 10 | part | exp(x)+log(x) |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 12 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))` | 4 | 13 | 8 | part | exp(x)+log(x) |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1)))` | 5 | 15 | 7 | part | exp(x)+log(x) |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 20 | full, certificate | log(log(x)), log(log(x)):compact_certificate |
| `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))` | 10 | 39 | 15 | full | exp(x)+log(x) |
| `E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 39 | 19 | certificate | exp(x)+log(x):compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 45 | 20 | full, certificate | x^2, x^2:compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 51 | 15 | full | x*log(x) |
| ... | ... | ... | ... | ... | 2 more macros in JSON |

### trial 1 / macro_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | log(x), x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 16 | certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | log(x), x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 15 | certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, 1), 1)))` | 4 | 9 | 8 | part | log(x) |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | log(x), x^2, x^2:compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(1, E(E(1, 1), 1))), 1)` | 5 | 11 | 7 | part | log(x) |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | log(x), x^2, x^2:compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 19 | certificate | log(log(x)):compact_certificate |
| `E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), 1)` | 10 | 21 | 15 | full | log(log(x)) |
| `E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 39 | 20 | full, certificate | exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, E(E(1, E(E(1, E(1, E(E(1, 1), 1))), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 51 | 15 | full | log(x) |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 45 | 20 | full, certificate | x^2, x^2:compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 15 | 51 | 20 | full, certificate | x*log(x), x*log(x):compact_certificate |
| ... | ... | ... | ... | ... | 1 more macros in JSON |

### trial 1 / kathara_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 13 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, 1), 1)` | 2 | 5 | 10 | part | exp(x)+log(x) |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 12 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))` | 4 | 13 | 8 | part | exp(x)+log(x) |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1)))` | 5 | 15 | 7 | part | exp(x)+log(x) |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 20 | full, certificate | log(log(x)), log(log(x)):compact_certificate |
| `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))` | 10 | 39 | 15 | full | exp(x)+log(x) |
| `E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 39 | 19 | certificate | exp(x)+log(x):compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 45 | 20 | full, certificate | x^2, x^2:compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 51 | 15 | full | x*log(x) |
| ... | ... | ... | ... | ... | 2 more macros in JSON |

### trial 2 / macro_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | exp(x), x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | exp(x), log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | certificate, certificate_part | exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(x, x)` | 1 | 3 | 8 | part | exp(x) |
| `E(1, E(1, 1))` | 2 | 5 | 10 | part | exp(x) |
| `E(1, E(1, x))` | 2 | 5 | 10 | part | exp(x) |
| `E(1, E(x, 1))` | 2 | 5 | 13 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | exp(x), x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, x), 1)` | 2 | 5 | 10 | part | exp(x) |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | exp(x), x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(1, 1)), 1)` | 3 | 7 | 9 | part | exp(x) |
| `E(E(1, E(1, x)), 1)` | 3 | 7 | 9 | part | exp(x) |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 12 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, x)), 1))` | 4 | 9 | 8 | part | exp(x) |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(E(1, E(E(1, x), 1)), 1), 1)` | 5 | 11 | 7 | part | exp(x)+log(x) |
| `E(E(1, E(E(1, E(1, x)), 1)), E(E(x, x), 1))` | 5 | 15 | 7 | part | exp(x) |
| ... | ... | ... | ... | ... | 8 more macros in JSON |

### trial 2 / kathara_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 16 | certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 15 | certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 20 | full, certificate | log(log(x)), log(log(x)):compact_certificate |
| `E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 39 | 20 | full, certificate | exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 45 | 20 | full, certificate | x^2, x^2:compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 51 | 15 | full | x*log(x) |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 15 | 51 | 19 | certificate | x*log(x):compact_certificate |
| `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 16 | 81 | 20 | full, certificate | x^2+log(x), x^2+log(x):compact_certificate |

### trial 3 / macro_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 16 | certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 15 | certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part, part | log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(E(1, x), 1)), 1))` | 5 | 11 | 7 | part | x*log(x) |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 39 | 20 | full, certificate | exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 45 | 20 | full, certificate | x^2, x^2:compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 15 | 51 | 19 | certificate | x*log(x):compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(E(1, E(E(1, x), 1)), 1)), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 16 | 59 | 15 | full | x*log(x) |
| `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 16 | 81 | 20 | full, certificate | x^2+log(x), x^2+log(x):compact_certificate |

### trial 3 / kathara_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 13 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, 1), 1)` | 2 | 5 | 10 | part | exp(x)+log(x) |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 12 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | certificate_part, part | log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, E(1, E(E(1, x), 1)))` | 4 | 9 | 8 | part | log(log(x)) |
| `E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))` | 4 | 13 | 8 | part | exp(x)+log(x) |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | certificate_part, part | log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 7 | part | log(log(x)) |
| `E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1)))` | 5 | 15 | 7 | part | exp(x)+log(x) |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 19 | certificate | log(log(x)):compact_certificate |
| `E(x, E(E(x, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 15 | full | log(log(x)) |
| `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))` | 10 | 39 | 15 | full | exp(x)+log(x) |
| ... | ... | ... | ... | ... | 5 more macros in JSON |

### trial 4 / macro_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 13 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 12 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | part, certificate_part | log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(E(1, E(E(1, x), 1)), 1), 1)` | 5 | 11 | 7 | part | exp(x)+log(x) |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 20 | full, certificate | log(log(x)), log(log(x)):compact_certificate |
| `E(E(1, E(E(1, E(x, 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 11 | 39 | 19 | certificate | exp(x)+log(x):compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 45 | 20 | full, certificate | x^2, x^2:compact_certificate |
| `E(E(E(1, E(E(1, E(1, E(E(1, E(1, E(E(1, x), 1))), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 12 | 51 | 15 | full | x*log(x) |
| `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(E(E(1, E(E(1, x), 1)), 1), 1), 1))), 1)), E(1, 1)), 1))` | 13 | 47 | 15 | full | exp(x)+log(x) |
| `E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), 1))), 1)), E(1, 1)), 1)), 1)` | 15 | 51 | 19 | certificate | x*log(x):compact_certificate |
| `E(E(1, E(E(1, E(E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1)), 1)), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))), 1)), E(1, 1)), 1))` | 16 | 81 | 20 | full, certificate | x^2+log(x), x^2+log(x):compact_certificate |

### trial 4 / kathara_mined

| expr | depth | size | priority | kinds | sources |
|---|---:|---:|---:|---|---|
| `1` | 0 | 1 | 1 | part, certificate_part | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `x` | 0 | 1 | 2 | part, certificate_part, full | exp(x), exp(x):compact_certificate, log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)), exp(log(x)):compact_certificate, log(exp(x)), log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, 1)` | 1 | 3 | 17 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, x)` | 1 | 3 | 17 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, 1)` | 1 | 3 | 20 | full, part, certificate, certificate_part | exp(x), exp(x):compact_certificate, log(exp(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate |
| `E(1, E(x, 1))` | 2 | 5 | 13 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, 1), 1)` | 2 | 5 | 19 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, x), 1)` | 2 | 5 | 19 | part, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, 1), 1)` | 2 | 5 | 10 | part | exp(x)+log(x) |
| `E(1, E(E(1, 1), 1))` | 3 | 7 | 18 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, x), 1))` | 3 | 7 | 20 | full, part, certificate, certificate_part | log(x), log(x):compact_certificate, log(log(x)), log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(log(x)):compact_certificate, exp(x)+log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(1, E(x, 1)), 1)` | 3 | 7 | 12 | certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(1, E(1, E(E(1, x), 1)))` | 4 | 9 | 17 | certificate_part, part | log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(x, 1)), 1))` | 4 | 9 | 20 | certificate, certificate_part | log(exp(x)):compact_certificate, exp(x)+log(x):compact_certificate |
| `E(E(1, E(E(1, x), 1)), 1)` | 4 | 9 | 20 | part, certificate_part, certificate | x^2, x^2:compact_certificate, x*log(x), exp(log(x)):compact_certificate, exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(x, E(1, E(E(1, x), 1)))` | 4 | 9 | 8 | part | log(log(x)) |
| `E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))` | 4 | 13 | 8 | part | exp(x)+log(x) |
| `E(E(1, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 16 | certificate_part, part | log(log(x)):compact_certificate, x^2, x^2:compact_certificate, x*log(x), x*log(x):compact_certificate, exp(x)+log(x), x^2+log(x), x^2+log(x):compact_certificate |
| `E(E(x, E(1, E(E(1, x), 1))), 1)` | 5 | 11 | 7 | part | log(log(x)) |
| `E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1)))` | 5 | 15 | 7 | part | exp(x)+log(x) |
| `E(E(1, E(E(1, 1), 1)), E(E(1, E(E(1, x), 1)), 1))` | 5 | 17 | 16 | part, certificate_part | x^2, x^2:compact_certificate, x*log(x), exp(x)+log(x):compact_certificate, x^2+log(x), x^2+log(x):compact_certificate |
| `E(1, E(E(1, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 19 | certificate | log(log(x)):compact_certificate |
| `E(x, E(E(x, E(1, E(E(1, x), 1))), 1))` | 6 | 13 | 15 | full | log(log(x)) |
| `E(E(1, E(E(1, E(1, E(E(1, x), 1))), 1)), E(E(E(1, E(E(1, E(E(1, E(E(1, 1), 1)), E(E(x, 1), 1))), 1)), E(1, 1)), 1))` | 10 | 39 | 15 | full | exp(x)+log(x) |
| ... | ... | ... | ... | ... | 5 more macros in JSON |

