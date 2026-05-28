
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ============================================================
# Configuration: Model column names in the CSV files
# ============================================================

# Maps short labels used in the paper to the actual column names in the CSVs.
# Update these if column names change in the data files.

CONFIGS = {
    # --- SFT-initialized + RL ---
    "Llama_SFT_Ravg": "vllmsftBase_average_gspo_merged_05.02__no_grammar",
    "Llama_SFT_R1":   "vllmgspo_merged_28.01__no_grammar",
    "Llama_SFT_R2":   "vllmgspo_merged_30.01__no_grammar",
    "Llama_SFT_R3":   "vllmLlama-3.1-8B-Instruct_sft_base_gspo_merged_28.01__no_grammar",
    "Llama_SFT_R4":   "vllmsft_base_pragmaticHigher_merged_04.02_cp80__no_grammar",
    "Llama_SFT_R5":   "vllmsftBase_semanticHigher_merged_04.02__no_grammar",

    "Qwen_SFT_Ravg":  "vllmsftBase_average_gspo_merged_09.03__no_grammar",
    "Qwen_SFT_R1":    "vllmsftBase_sameWeight_gspo_merged_09.03__no_grammar",
    "Qwen_SFT_R2":    "vllmsftBase_noNeg_merged_09.03__no_grammar",
    "Qwen_SFT_R3":    "vllmsftBase_syntacticHigher_gspo_merged_10.03__no_grammar",
    "Qwen_SFT_R4":    "vllmsftBase_pragmaticHigher_gspo_merged_12.03__no_grammar",
    "Qwen_SFT_R5":    "vllmsftBase_semanticHigher_gspo_merged_12.03__no_grammar",

    # --- Untrained (BPM-unadapted) + RL ---
    "Llama_Unt_R1":   "vllmuntrained_sameWeight_gspo_merged_28.01__no_grammar",
    "Qwen_Unt_R1":    "vllmsameWeight_gspo_merged_05.03__no_grammar",

    # --- SFT-only baselines (no RL) ---
    "Llama_SFTonly":   "vllmllama-3.1-8b-sft-merged",
    "Qwen_SFTonly":    "vllmqwen2.5-14b-lora-merged",
}


# ============================================================
# Test definitions: (RQ, Model, A_label, B_label, Bonferroni_m, A_name, B_name)
# ============================================================

TESTS = [
    # RQ1: GSPO vs SFT-only
    ("RQ1", "Llama", "Llama_SFT_R1",   "Llama_SFTonly",  3, "SFT+R1",      "SFT-only"),
    ("RQ1", "Qwen",  "Qwen_SFT_R1",    "Qwen_SFTonly",   3, "SFT+R1",      "SFT-only"),

    # RQ2a: Reward weighting (R1 vs R3, R4, R5)
    ("RQ2a", "Llama", "Llama_SFT_R1",  "Llama_SFT_R3",   9, "R1",          "R3(syn)"),
    ("RQ2a", "Llama", "Llama_SFT_R1",  "Llama_SFT_R4",   9, "R1",          "R4(pra)"),
    ("RQ2a", "Llama", "Llama_SFT_R1",  "Llama_SFT_R5",   9, "R1",          "R5(sem)"),
    ("RQ2a", "Qwen",  "Qwen_SFT_R1",   "Qwen_SFT_R3",   9, "R1",          "R3(syn)"),
    ("RQ2a", "Qwen",  "Qwen_SFT_R1",   "Qwen_SFT_R4",   9, "R1",          "R4(pra)"),
    ("RQ2a", "Qwen",  "Qwen_SFT_R1",   "Qwen_SFT_R5",   9, "R1",          "R5(sem)"),

    # RQ2b: Invalidity penalty (R1 vs R2)
    ("RQ2b", "Llama", "Llama_SFT_R1",  "Llama_SFT_R2",   3, "R1(p=-1)",    "R2(p=0)"),
    ("RQ2b", "Qwen",  "Qwen_SFT_R1",   "Qwen_SFT_R2",   3, "R1(p=-1)",    "R2(p=0)"),

    # RQ3: SFT-initialized vs BPM-unadapted base
    ("RQ3",  "Llama", "Llama_SFT_R1",  "Llama_Unt_R1",   3, "SFT+R1",      "Unt+R1"),
    ("RQ3",  "Qwen",  "Qwen_SFT_R1",   "Qwen_Unt_R1",   3, "SFT+R1",      "Unt+R1"),
]

# Exploratory comparison (not in the main test table)
EXPLORATORY_TESTS = [
    ("Expl", "Llama", "Llama_SFT_Ravg", "Llama_SFT_R1",  3, "Ravg",        "R1"),
]

# Number of permutation resamples
N_RESAMPLES = 99_999

# Quality dimensions
DIMENSIONS = ["syn", "pra", "sem"]
DIM_NAMES = {"syn": "Syntactic", "pra": "Pragmatic", "sem": "Semantic"}


# ============================================================
# Data loading
# ============================================================

def load_csv(filepath: Path) -> Dict[str, List[Optional[float]]]:
    """Load a semicolon-delimited CSV and return {column_name: [values]}."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        data = {col: [] for col in header[1:]}  # skip first column (bpmn id)
        for row in reader:
            for i, col in enumerate(header[1:], 1):
                val = row[i].strip() if i < len(row) else ""
                data[col].append(float(val) if val else None)
    return data


def load_all_data(data_dir: Path) -> Dict[str, Dict[str, Dict[str, List[Optional[float]]]]]:
    """
    Load all CSV files into a nested dict:
      data[dim]['rl' | 'sft'][column_name] = [values]
    """
    data = {}
    for dim in DIMENSIONS:
        dim_full = {"syn": "syntactic", "pra": "pragmatic", "sem": "semantic"}[dim]
        data[dim] = {
            "rl":  load_csv(data_dir / f"RL_{dim_full}_quality.csv"),
            "sft": load_csv(data_dir / f"sft_{dim_full}_quality.csv"),
        }
    return data


def get_scores(
    data: Dict, label: str, dim: str
) -> List[Optional[float]]:
    """Retrieve per-sample scores for a given configuration and dimension."""
    col = CONFIGS[label]
    if col in data[dim]["rl"]:
        return data[dim]["rl"][col]
    if col in data[dim]["sft"]:
        return data[dim]["sft"][col]
    raise KeyError(
        f"Column '{col}' (label '{label}') not found in {dim} data. "
        f"Available RL columns: {list(data[dim]['rl'].keys())[:5]}... "
        f"Available SFT columns: {list(data[dim]['sft'].keys())[:5]}..."
    )


# ============================================================
# Statistical testing
# ============================================================

def paired_permutation_test(
    a_scores: List[Optional[float]],
    b_scores: List[Optional[float]],
    n_resamples: int = N_RESAMPLES,
) -> Tuple[float, float, int, float]:
    """
    Two-sided paired permutation test (sign-flip variant).

    Only samples where both configurations produced valid output are included
    (complete-case analysis conditioned on joint validity).

    Parameters
    ----------
    a_scores : list of float or None
        Per-sample quality scores for configuration A.
    b_scores : list of float or None
        Per-sample quality scores for configuration B.
    n_resamples : int
        Number of permutation resamples.

    Returns
    -------
    delta : float
        Mean paired difference (A - B).
    p_value : float
        Two-sided permutation p-value (unadjusted).
    n_pairs : int
        Number of valid paired samples.
    skewness : float
        Skewness of the paired differences (for diagnostics).

    References
    ----------
    Good, P. I. (2005). Permutation, Parametric, and Bootstrap Tests of
    Hypotheses (3rd ed.). Springer.
    """
    # Form pairs where both are valid
    pairs = [
        (a, b)
        for a, b in zip(a_scores, b_scores)
        if a is not None and b is not None
    ]

    if len(pairs) < 10:
        return np.nan, np.nan, len(pairs), np.nan

    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])

    # Test statistic: mean paired difference
    def stat_func(x, y, axis):
        return np.mean(x - y, axis=axis)

    result = stats.permutation_test(
        (a_arr, b_arr),
        stat_func,
        n_resamples=n_resamples,
        permutation_type="samples",
        alternative="two-sided",
    )

    delta = np.mean(a_arr - b_arr)
    skewness = float(stats.skew(a_arr - b_arr))

    return delta, result.pvalue, len(pairs), skewness


def significance_stars(p_adj: float) -> str:
    """Return significance notation."""
    if np.isnan(p_adj):
        return "n/a"
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return "n.s."


# ============================================================
# Descriptive statistics
# ============================================================

def compute_descriptives(
    data: Dict, label: str
) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, and n for each dimension."""
    result = {}
    for dim in DIMENSIONS:
        scores = get_scores(data, label, dim)
        valid = [s for s in scores if s is not None]
        result[dim] = {
            "mean": np.mean(valid) if valid else np.nan,
            "std":  np.std(valid, ddof=1) if len(valid) > 1 else np.nan,
            "n":    len(valid),
        }
    return result


# ============================================================
# Main analysis
# ============================================================

def run_analysis(data_dir: Path, output_file: Optional[Path] = None, n_resamples: int = N_RESAMPLES):
    """Run all statistical tests and print results."""
    print("=" * 100)
    print("STATISTICAL ANALYSIS: Paired Permutation Tests")
    print(f"Resamples: {n_resamples:,}")
    print(f"Minimum attainable p-value: {2 / (n_resamples + 1):.1e}")
    print(f"Data directory: {data_dir}")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    data = load_all_data(data_dir)
    print("Data loaded successfully.\n")

    # -------------------------------------------------------
    # Part 1: Descriptive statistics
    # -------------------------------------------------------
    print("=" * 100)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 100)
    print(f"\n{'Label':<22} {'Syn Mean':>8} {'Syn Std':>8} {'Pra Mean':>8} "
          f"{'Pra Std':>8} {'Sem Mean':>8} {'Sem Std':>8} {'n':>4}")
    print("-" * 85)

    for label in CONFIGS:
        try:
            desc = compute_descriptives(data, label)
            print(
                f"{label:<22} "
                f"{desc['syn']['mean']:8.4f} {desc['syn']['std']:8.4f} "
                f"{desc['pra']['mean']:8.4f} {desc['pra']['std']:8.4f} "
                f"{desc['sem']['mean']:8.4f} {desc['sem']['std']:8.4f} "
                f"{desc['syn']['n']:4d}"
            )
        except KeyError as e:
            print(f"{label:<22} ERROR: {e}")

    # -------------------------------------------------------
    # Part 2: Symmetry diagnostics (justifying permutation over Wilcoxon)
    # -------------------------------------------------------
    print("\n" + "=" * 100)
    print("SYMMETRY DIAGNOSTICS (justification for permutation test over Wilcoxon)")
    print("Wilcoxon signed-rank test requires symmetric difference distributions.")
    print("|skew| > 1 indicates potential symmetry violation.")
    print("=" * 100)

    for rq, model, a_key, b_key, bonf, a_name, b_name in TESTS:
        for dim in DIMENSIONS:
            a_raw = get_scores(data, a_key, dim)
            b_raw = get_scores(data, b_key, dim)
            pairs = [
                (a, b) for a, b in zip(a_raw, b_raw)
                if a is not None and b is not None
            ]
            if len(pairs) < 10:
                continue
            diffs = np.array([a - b for a, b in pairs])
            skew = stats.skew(diffs)
            flag = " *** VIOLATION" if abs(skew) > 1 else ""
            if abs(skew) > 0.5:
                print(
                    f"  {rq:5s} {model:5s} {a_name:10s} vs {b_name:10s} "
                    f"{DIM_NAMES[dim]:9s}: skew={skew:+.3f} (n={len(pairs)}){flag}"
                )

    # -------------------------------------------------------
    # Part 3: Main permutation tests
    # -------------------------------------------------------
    print("\n" + "=" * 100)
    print("MAIN PERMUTATION TESTS (Bonferroni-adjusted)")
    print("=" * 100)

    header = (
        f"\n{'RQ':<5} {'Model':<6} {'A':>10} {'B':>10} {'m':>2} "
        f"{'Dim':>4} {'Delta':>8} {'p_raw':>12} {'p_adj':>12} {'Sig':>4} {'n':>4} {'Skew':>6}"
    )
    print(header)
    print("-" * 95)

    all_results = []

    for rq, model, a_key, b_key, bonf, a_name, b_name in TESTS:
        for dim in DIMENSIONS:
            a_scores = get_scores(data, a_key, dim)
            b_scores = get_scores(data, b_key, dim)

            delta, p_raw, n_pairs, skew = paired_permutation_test(
                a_scores, b_scores, n_resamples
            )

            p_adj = min(p_raw * bonf, 1.0) if not np.isnan(p_raw) else np.nan
            sig = significance_stars(p_adj)

            result = {
                "rq": rq, "model": model,
                "a_name": a_name, "b_name": b_name,
                "bonf": bonf, "dim": dim,
                "delta": delta, "p_raw": p_raw,
                "p_adj": p_adj, "sig": sig,
                "n": n_pairs, "skew": skew,
            }
            all_results.append(result)

            print(
                f"{rq:<5} {model:<6} {a_name:>10} {b_name:>10} {bonf:>2} "
                f"{DIM_NAMES[dim]:>4} {delta:>+8.4f} {p_raw:>12.2e} "
                f"{p_adj:>12.2e} {sig:>4} {n_pairs:>4} {skew:>+6.2f}"
            )
        print()

    # -------------------------------------------------------
    # Part 4: Exploratory tests (R_avg vs R1)
    # -------------------------------------------------------
    print("\n" + "=" * 100)
    print("EXPLORATORY TESTS (interpreted cautiously)")
    print("R_avg and R1 are mathematically equivalent; differences may reflect")
    print("implementation-level effects or stochastic training variability.")
    print("=" * 100)
    print(header)
    print("-" * 95)

    for rq, model, a_key, b_key, bonf, a_name, b_name in EXPLORATORY_TESTS:
        for dim in DIMENSIONS:
            a_scores = get_scores(data, a_key, dim)
            b_scores = get_scores(data, b_key, dim)

            delta, p_raw, n_pairs, skew = paired_permutation_test(
                a_scores, b_scores, n_resamples
            )

            p_adj = min(p_raw * bonf, 1.0) if not np.isnan(p_raw) else np.nan
            sig = significance_stars(p_adj)

            print(
                f"{rq:<5} {model:<6} {a_name:>10} {b_name:>10} {bonf:>2} "
                f"{DIM_NAMES[dim]:>4} {delta:>+8.4f} {p_raw:>12.2e} "
                f"{p_adj:>12.2e} {sig:>4} {n_pairs:>4} {skew:>+6.2f}"
            )

    # -------------------------------------------------------
    # Part 5: Comparison with Wilcoxon (robustness check)
    # -------------------------------------------------------
    print("\n" + "=" * 100)
    print("ROBUSTNESS CHECK: Permutation vs Wilcoxon Signed-Rank")
    print("Disagreements indicate cases where Wilcoxon's symmetry assumption")
    print("affects the conclusion (significance level changes at alpha=0.05).")
    print("=" * 100)

    disagreements = 0
    total = 0

    for rq, model, a_key, b_key, bonf, a_name, b_name in TESTS:
        for dim in DIMENSIONS:
            a_scores = get_scores(data, a_key, dim)
            b_scores = get_scores(data, b_key, dim)

            # Permutation test
            delta, p_perm, n_pairs, skew = paired_permutation_test(
                a_scores, b_scores, n_resamples
            )
            if np.isnan(p_perm):
                continue

            p_perm_adj = min(p_perm * bonf, 1.0)

            # Wilcoxon signed-rank test
            pairs = [
                (a, b) for a, b in zip(a_scores, b_scores)
                if a is not None and b is not None
            ]
            a_arr = np.array([p[0] for p in pairs])
            b_arr = np.array([p[1] for p in pairs])

            try:
                _, p_wilc = stats.wilcoxon(a_arr, b_arr)
                p_wilc_adj = min(p_wilc * bonf, 1.0)
            except ValueError:
                continue

            total += 1
            perm_sig = p_perm_adj < 0.05
            wilc_sig = p_wilc_adj < 0.05

            if perm_sig != wilc_sig:
                disagreements += 1
                print(
                    f"  DISAGREE: {rq} {model} {a_name} vs {b_name} [{DIM_NAMES[dim]}]: "
                    f"Perm p_adj={p_perm_adj:.2e} ({'sig' if perm_sig else 'n.s.'}), "
                    f"Wilcoxon p_adj={p_wilc_adj:.2e} ({'sig' if wilc_sig else 'n.s.'}), "
                    f"skew={skew:+.2f}"
                )

    print(f"\n  {disagreements} disagreements out of {total} comparisons")
    if disagreements == 0:
        print("  All tests agree on significance at alpha=0.05.")

    # -------------------------------------------------------
    # Part 6: LaTeX table output
    # -------------------------------------------------------
    print("\n" + "=" * 100)
    print("LATEX TABLE OUTPUT (for Table 5 in the paper)")
    print("=" * 100)

    latex_lines = []
    latex_lines.append(r"\begin{table*}[]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        r"\caption{Consolidated pairwise permutation tests "
        r"(Bonferroni-adjusted, " + f"{n_resamples:,}" + r" resamples). "
        r"All comparisons use the no-grammar condition. "
        r"$\Delta$ is the mean paired difference $A - B$. "
        r"Significance: \textsuperscript{***}$p < 0.001$, "
        r"\textsuperscript{**}$p < 0.01$, "
        r"\textsuperscript{*}$p < 0.05$, n.s.\ = not significant. "
        r"``Unt'' denotes the untrained (BPM-unadapted instruction-tuned) base.}"
    )
    latex_lines.append(r"\label{tab:stat_tests}")
    latex_lines.append(r"\smallskip")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(
        r"\begin{tabular}{ll ll c rrr rrr rrr}"
    )
    latex_lines.append(r"\toprule")
    latex_lines.append(
        r"& & & & & \multicolumn{3}{c}{\textbf{Syntactic}} "
        r"& \multicolumn{3}{c}{\textbf{Pragmatic}} "
        r"& \multicolumn{3}{c}{\textbf{Semantic}} \\"
    )
    latex_lines.append(
        r"\cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}"
    )
    latex_lines.append(
        r"RQ & Model & $A$ & $B$ & $m$ "
        r"& $\Delta$ & $p_{\mathrm{adj}}$ & "
        r"& $\Delta$ & $p_{\mathrm{adj}}$ & "
        r"& $\Delta$ & $p_{\mathrm{adj}}$ & \\"
    )
    latex_lines.append(r"\midrule")

    # Group results by test
    for i, (rq, model, a_key, b_key, bonf, a_name, b_name) in enumerate(TESTS):
        # Get results for all 3 dimensions
        test_results = [
            r for r in all_results
            if r["rq"] == rq and r["model"] == model
            and r["a_name"] == a_name and r["b_name"] == b_name
        ]

        def fmt_delta(d):
            sign = "+" if d >= 0 else "-"
            return f"${sign}${abs(d):.3f}"

        def fmt_p(p):
            if p >= 1.0:
                return "$1.00$"
            exp = int(np.floor(np.log10(p)))
            coeff = p / (10 ** exp)
            return f"${coeff:.1f} \\times 10^{{{exp}}}$"

        def fmt_sig(s):
            if s == "***":
                return r"\textsuperscript{***}"
            if s == "**":
                return r"\textsuperscript{**}"
            if s == "*":
                return r"\textsuperscript{*}"
            return "n.s."

        syn = next((r for r in test_results if r["dim"] == "syn"), None)
        pra = next((r for r in test_results if r["dim"] == "pra"), None)
        sem = next((r for r in test_results if r["dim"] == "sem"), None)

        if syn and pra and sem:
            # Determine multirow
            rq_cell = rq
            line = (
                f" & {model} & {a_name} & {b_name} & {bonf} "
                f"& {fmt_delta(syn['delta'])} & {fmt_p(syn['p_adj'])} & {fmt_sig(syn['sig'])} "
                f"& {fmt_delta(pra['delta'])} & {fmt_p(pra['p_adj'])} & {fmt_sig(pra['sig'])} "
                f"& {fmt_delta(sem['delta'])} & {fmt_p(sem['p_adj'])} & {fmt_sig(sem['sig'])} \\\\"
            )
            latex_lines.append(line)

        # Add midrule between RQ groups
        if i < len(TESTS) - 1 and TESTS[i + 1][0] != rq:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}%")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table*}")

    latex_output = "\n".join(latex_lines)
    print(latex_output)

    # Save to file if requested
    if output_file:
        output_file.write_text(latex_output, encoding="utf-8")
        print(f"\nLaTeX table saved to: {output_file}")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis for the RL reward design paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the CSV data files (default: current directory)",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=None,
        help="Optional: save LaTeX table to this file",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=N_RESAMPLES,
        help=f"Number of permutation resamples (default: {N_RESAMPLES:,})",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Check that required files exist
    required_files = []
    for dim in ["syntactic", "pragmatic", "semantic"]:
        for prefix in ["RL", "sft"]:
            required_files.append(f"{prefix}_{dim}_quality.csv")

    missing = [f for f in required_files if not (args.data_dir / f).exists()]
    if missing:
        print(f"Error: Missing data files in {args.data_dir}:", file=sys.stderr)
        for f in missing:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    run_analysis(args.data_dir, args.output_latex, args.resamples)


if __name__ == "__main__":
    main()
