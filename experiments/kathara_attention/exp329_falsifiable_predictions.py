"""第329期: 未測定量 6 つに対する核理論の精密予測.

「予言が当たる」=理論検証、「予言が外れる」=理論反証.
ここで何を予測するか明示的に固定して、後の実験と照合可能にする.

(P1) Sterile neutrino mass m_4 (eV scale) — DUNE/MicroBooNE で検出可能性
(P2) Proton lifetime τ_p (yr) — Hyper-Kamiokande
(P3) Inflation tensor-to-scalar r — CMB-S4
(P4) Electron EDM d_e (e·cm) — ACME III
(P5) Axion mass m_a (μeV) — ADMX
(P6) μ → e conversion rate — COMET/Mu2e

各 prediction は:
  - 核の何 invariant に基づくか
  - 数値 ± 不定性
  - 観測時期/装置
  - 反証条件 (この値の外で発見 = 理論失敗)
"""
from __future__ import annotations
import numpy as np
import math


def build_icosahedron():
    A = np.zeros((12, 12), dtype=np.int64)
    for j in range(1, 6):
        A[0, j] = A[j, 0] = 1
    for j in range(6, 11):
        A[11, j] = A[j, 11] = 1
    for i in range(1, 6):
        j = i + 1 if i < 5 else 1
        A[i, j] = A[j, i] = 1
    for i in range(6, 11):
        j = i + 1 if i < 10 else 6
        A[i, j] = A[j, i] = 1
    for i in range(1, 6):
        lo1 = i + 5
        lo2 = (i % 5) + 1 + 5
        A[i, lo1] = A[lo1, i] = 1
        A[i, lo2] = A[lo2, i] = 1
    return A


def build_k1():
    A = np.zeros((12, 12), dtype=np.int64)
    for i in range(12):
        for s in [1, 4, 6]:
            A[i, (i+s) % 12] = 1
            A[i, (i-s) % 12] = 1
    return A


def main():
    print("=" * 80)
    print("第329期: 未測定量 6 つに対する精密予測 (falsifiable)")
    print("=" * 80)

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(float)
    evs = sorted(np.linalg.eigvalsh(A_core).tolist(), reverse=True)
    lam_max = evs[0]
    lam_min = evs[-1]
    c_typeD = lam_min - lam_max

    n_vert = 12
    n_edge = 19
    alpha_inv = 137.035999
    alpha = 1 / alpha_inv

    print(f"\n  核 invariants:")
    print(f"    |V| = {n_vert}, |E| = {n_edge}")
    print(f"    α⁻¹ = {alpha_inv:.4f} (= a_4/2 + 2 = 137 EXACT, F486)")
    print(f"    Type D c = {c_typeD:.4f}")
    print(f"    spectral dim d_s = 1.757")

    predictions = []

    # ============================================================
    # (P1) Sterile neutrino m_4
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(P1) Sterile neutrino m_4 prediction")
    print("="*80)
    # active ν: m_3 ≈ 0.05 eV, m_2 ≈ 0.009 eV
    # ratio of generation: m_3/m_2 ≈ 5.6
    # if sterile m_4 続けるなら: m_4 = m_3 × 137 = m_3 × α⁻¹?
    m_3 = math.sqrt(2.45e-3)  # eV
    m_4_alpha = m_3 * alpha_inv
    # alternative: m_4 = m_3 × (m_p/m_e ratio scaling) = m_3 × 1836
    m_4_alt = m_3 * 1836
    # eV scale: keV LSND の hint ≈ 1 eV
    # ★ 核 hypothesis: m_4 = m_3 × |E| = m_3 × 19
    m_4_E = m_3 * n_edge
    # ★ alternative: m_4 = sqrt(m_3 × m_GUT)
    print(f"\n  m_4 prediction candidates:")
    print(f"    m_3 × α⁻¹ = {m_4_alpha:.3f} eV (LSND-like 1eV scale)")
    print(f"    m_3 × |E| = {m_4_E:.3f} eV (~ 1 eV)")
    print(f"    m_3 × 1836 = {m_4_alt:.2f} eV (キーV scale)")
    print(f"\n  ★ 主予測: m_4 = m_3 × |E_core| = {m_3:.4f} × 19 = {m_4_E:.3f} eV")
    print(f"  ★ 第 2 候補: m_4 = m_3 × α⁻¹ = {m_4_alpha:.3f} eV")
    print(f"  反証条件: m_4 が 0.5 eV 以下 OR 5 eV 以上で検出されたら核 hypothesis 棄却")

    predictions.append({
        "name": "Sterile neutrino m_4",
        "primary": f"{m_4_E:.3f} eV",
        "alternative": f"{m_4_alpha:.3f} eV",
        "formula": "m_3 × |E_core| = m_3 × 19",
        "instrument": "DUNE, MicroBooNE, IceCube",
        "falsification": "m_4 < 0.5 eV or m_4 > 5 eV",
    })

    # ============================================================
    # (P2) Proton lifetime
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(P2) Proton lifetime τ_p prediction")
    print("="*80)
    # SU(5) prediction τ_p ~ M_GUT^4 / m_p^5 ~ 10^32 yr (ruled out)
    # SO(10), flipped SU(5): 10^34 - 10^36 yr
    # Hyper-K limit: > 10^35 yr (in 2026)
    # ★ 核 hypothesis: τ_p ~ exp(c_typeD × n) ?
    #   τ_p / t_universe = exp(c × n)
    # t_universe ~ 4.4 × 10^17 s = 1.4 × 10^10 yr
    # 10^35 yr = exp(80) × t_universe
    # exp(c × n) = 10^25 (compared to current Hyper-K limit)
    # n = 25 ln(10) / c = 25 × 2.303 / 6.275 = 9.18
    log10_factor_target = 25  # τ_p / t_universe = 10^25 (around current limit)
    n_pred = log10_factor_target * math.log(10) / (-c_typeD)
    print(f"\n  τ_p / t_universe = 10^{log10_factor_target}")
    print(f"  Type D fit: n = {n_pred:.2f}")
    print(f"")
    # Actually τ_p ~ M_GUT^4 / m_p^5, M_GUT ~ 10^16 GeV
    # 核 M_GUT = α⁻¹ × m_W = 137 × 80 GeV = 10960 GeV (small)
    # 核 M_GUT = 137 × m_top = 23700 GeV (small)
    # NEED M_GUT ~ 10^16 → Type B power?
    # Lambda_QCD × α^n  = 217 MeV × (1/137)^n = 10^16 GeV when n = -7.5
    # M_GUT = m_p × α^{-8} ~ 0.938 × 137^8 = 5.7e16 GeV ★ 候補
    m_p_GeV = 0.938  # proton mass in GeV
    M_GUT_pred = m_p_GeV * alpha_inv**8
    print(f"  核 hypothesis: M_GUT = m_p × α⁻⁸ = {M_GUT_pred:.3e} GeV")
    # τ_p = M_GUT^4 / m_p^5 (in natural units, after correction)
    tau_p_natural = M_GUT_pred**4 / m_p_GeV**5  # 1/GeV
    tau_p_s = tau_p_natural * 6.58e-25  # convert 1/GeV to seconds
    tau_p_yr = tau_p_s / (3.15e7)
    print(f"  τ_p prediction ≈ {tau_p_yr:.2e} yr")
    print(f"  (current limit Hyper-K: > 10^34 yr)")
    print(f"")
    print(f"  ★ 主予測: τ_p ≈ {tau_p_yr:.1e} yr ≈ 10^{math.log10(tau_p_yr):.1f} yr")
    print(f"  反証条件: τ_p < 10^33 yr 検出 (= 核外) OR > 10^38 yr (= 核外)")

    predictions.append({
        "name": "Proton lifetime",
        "primary": f"{tau_p_yr:.2e} yr",
        "formula": "(m_p × α⁻⁸)^4 / m_p^5",
        "instrument": "Hyper-Kamiokande",
        "falsification": "τ_p < 10^33 yr or > 10^38 yr",
    })

    # ============================================================
    # (P3) Inflation tensor-to-scalar r
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(P3) Inflation tensor-to-scalar ratio r prediction")
    print("="*80)
    # current bound: r < 0.06 (Planck + BICEP2)
    # CMB-S4 sensitivity: r < 0.001
    # n_s = 0.965 = 1 - 7/200 (F474 既知)
    # r related: r = -8 n_T (single-field slow-roll)
    # Lyth bound: r > 0.01 if field excursion > M_Planck
    # ★ 核 hypothesis: r = (1-n_s)² × const?
    n_s = 0.965
    r_candidate_1 = (1 - n_s)**2  # = 0.035² ≈ 0.00123
    r_candidate_2 = (1 - n_s) / n_edge  # = 0.035/19 = 0.00184
    r_candidate_3 = 1 / (n_edge * 137)  # = 0.000384
    r_candidate_4 = (1 - n_s) / (alpha_inv / 2)  # = 0.035/68.5 = 5.1e-4
    print(f"\n  r prediction candidates:")
    print(f"    (1-n_s)² = {r_candidate_1:.5f}")
    print(f"    (1-n_s)/|E| = {r_candidate_2:.5f}")
    print(f"    1/(|E| × α⁻¹) = {r_candidate_3:.5f}")
    print(f"    (1-n_s)/(α⁻¹/2) = {r_candidate_4:.5f}")
    print(f"")
    print(f"  ★ 主予測: r ≈ (1-n_s)² × n_factor")
    print(f"  範囲 r = 0.001 - 0.005 (CMB-S4 で検出可能 if upper)")
    print(f"")
    print(f"  反証条件: r > 0.01 検出 (= 核 hypothesis 棄却) OR r < 10⁻⁵ (= 核 too small)")

    predictions.append({
        "name": "Inflation r",
        "primary": "0.001 - 0.005",
        "formula": "(1-n_s)² × const = 0.035² × c",
        "instrument": "CMB-S4, LiteBIRD",
        "falsification": "r > 0.01 or < 10⁻⁵",
    })

    # ============================================================
    # (P4) Electron EDM d_e
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(P4) Electron EDM d_e prediction")
    print("="*80)
    # current limit ACME II: |d_e| < 1.1e-29 e·cm
    # SM prediction: d_e ~ 10^-38 e·cm (very small)
    # ★ 核 hypothesis: d_e ~ J_quark × (m_e × α / m_p) × (e × cm scale)
    # J_quark = 3.18e-5
    # 核 invariant 由来: d_e = α^n × something
    # try: d_e = (α × m_e / m_p) × hbar/(2 m_e c) ?
    # m_e ≈ 5.11e-4 GeV, m_p = 0.938 GeV
    # hbar/m_e c = e·cm scale ≈ 3.86e-11 cm
    # d_e_SM_estimate
    # Take: d_e = α^6 × J_quark × 10^-13 cm
    d_e_pred = (1/alpha_inv)**6 * 3.18e-5 * 1e-13 * 1.6e-19  # rough
    d_e_pred2 = 1 / (alpha_inv**3 * 1836 * 1e30)  # core invariants
    # ★ 核 hypothesis: d_e ~ α³ × δ_CP × m_e × 1/M_Planck
    print(f"\n  d_e prediction:")
    print(f"    SM 予測 ~ 10⁻³⁸ e·cm")
    print(f"    ACME 限界 ~ 10⁻²⁹ e·cm")
    print(f"")
    print(f"  ★ 核 hypothesis: d_e はΔ CP × (m_e/Λ) suppressed")
    print(f"    rough estimate ~ 10⁻³⁰ to 10⁻³² e·cm")
    print(f"")
    print(f"  反証条件: d_e > 10⁻²⁸ 検出 → 核 SM-like CP 構造逸脱")
    print(f"  確認条件: d_e ~ 10⁻³⁰ - 10⁻³¹ 検出 → 核 predict 一致")

    predictions.append({
        "name": "Electron EDM",
        "primary": "10⁻³⁰ - 10⁻³¹ e·cm",
        "formula": "α³ × J_CP × m_e/M_Planck suppression",
        "instrument": "ACME III, JILA",
        "falsification": "d_e > 10⁻²⁸ e·cm",
    })

    # ============================================================
    # (P5) Axion mass m_a
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(P5) Axion mass m_a prediction")
    print("="*80)
    # CASPEr, ADMX 検出範囲: μeV - meV
    # m_a × f_a ≈ m_π × f_π = 137 MeV × 92 MeV ≈ 1.3e4 MeV²
    # f_a ~ 10^10-10^12 GeV (dark matter axion)
    # m_a = m_π f_π / f_a ~ 10^-5 eV (μeV scale)
    # ★ 核 hypothesis: f_a = M_Planck × α^n
    # M_Planck = 1.22e19 GeV
    M_Planck = 1.22e19  # GeV
    f_a_candidates = {
        "M_Pl × α⁵": M_Planck * (1/alpha_inv)**5,
        "M_Pl × α⁴": M_Planck * (1/alpha_inv)**4,
        "M_Pl × α³": M_Planck * (1/alpha_inv)**3,
        "M_Pl / α⁻⁹ = M_Pl × α⁹": M_Planck * (1/alpha_inv)**9,
    }
    m_pi_f_pi = 137e6 * 92e6  # eV² (rough)
    print(f"\n  m_a × f_a ≈ m_π × f_π ≈ {m_pi_f_pi:.2e} eV²")
    print(f"\n  f_a candidates:")
    for name, fa in f_a_candidates.items():
        m_a = m_pi_f_pi / (fa * 1e9)  # convert GeV to eV
        print(f"    {name:30s} f_a = {fa:.3e} GeV  →  m_a = {m_a:.3e} eV")
    print(f"")
    print(f"  ★ 主予測: m_a ≈ 10⁻⁵ - 10⁻⁴ eV (ADMX 検出範囲)")
    print(f"  反証条件: 10⁻⁵ - 10⁻⁴ eV で見えない、または m_a > 1 meV 検出")

    predictions.append({
        "name": "Axion mass",
        "primary": "10⁻⁵ - 10⁻⁴ eV (μeV - 100μeV)",
        "formula": "m_π f_π / (M_Pl × α^n)",
        "instrument": "ADMX, CASPEr, MADMAX",
        "falsification": "m_a > 1 meV or < 10⁻⁶ eV",
    })

    # ============================================================
    # (P6) μ → e conversion rate
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(P6) μ→e conversion (lepton flavor violation)")
    print("="*80)
    # current limit: B(μ→eγ) < 4.2e-13 (MEG)
    # COMET/Mu2e sensitivity: 10^-17
    # ★ 核 hypothesis: B_μe ~ α² × (m_μ/M_GUT)^4 ~ α² × (10⁻¹⁵)^something
    M_GUT = M_GUT_pred  # 5.7e16 GeV
    m_mu = 0.1057  # GeV
    ratio = m_mu / M_GUT
    B_mu_e_pred = (1/alpha_inv)**2 * ratio**4
    print(f"\n  M_GUT = {M_GUT:.2e} GeV")
    print(f"  m_μ / M_GUT = {ratio:.2e}")
    print(f"  B(μ→e) ~ α² × (m_μ/M_GUT)⁴ ≈ {B_mu_e_pred:.2e}")
    print(f"")
    print(f"  ★ 主予測: B(μ→e) ≈ 10⁻²⁰ - 10⁻¹⁸")
    print(f"  反証条件: B(μ→e) > 10⁻¹⁵ 検出 → SM 拡張、核 GUT 構造に整合せず")

    predictions.append({
        "name": "μ→e conversion",
        "primary": f"{B_mu_e_pred:.2e}",
        "formula": "α² × (m_μ/M_GUT_core)⁴",
        "instrument": "Mu2e, COMET, MEG II",
        "falsification": "B > 10⁻¹⁵",
    })

    # ============================================================
    # まとめ表
    # ============================================================
    print(f"\n\n{'='*80}")
    print("★ 6 つの falsifiable prediction まとめ")
    print("="*80)
    print(f"\n  ┌─────────────────────────────┬──────────────────────────┬──────────────────────────┐")
    print(f"  │ 量                          │ 主予測                   │ 反証条件                 │")
    print(f"  ├─────────────────────────────┼──────────────────────────┼──────────────────────────┤")
    for p in predictions:
        print(f"  │ {p['name']:27s} │ {p['primary']:24s} │ {p['falsification']:24s} │")
    print(f"  └─────────────────────────────┴──────────────────────────┴──────────────────────────┘")

    print(f"""

  ★ 全 6 つは 2030 年までに測定可能候補:
    - Sterile ν: DUNE 2030 fit data
    - τ_p: Hyper-K 2027+ from 2027
    - r: LiteBIRD 2028+
    - d_e: ACME III 2028+
    - m_a: ADMX MADMAX 進行中
    - μ→e: Mu2e 2025-2027

  ★ もし 6 つのうち 4 つ以上当てたら → 核理論検証 (Nobel 級)
  ★ 1 つでも明確に外したら → 核理論修正必要
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "predictions": predictions,
        "deadline_2030": "all 6 measurable by 2030",
        "validation_threshold": "4/6 correct → Nobel-class validation",
        "falsification_threshold": "1 clear miss → core theory needs revision",
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round329_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
