#!/usr/bin/env python3
"""Generate Paper/tables/tab_all_results.tex from results.json + changed_cells.json.

Cell key format in changed_cells.json:  "dataset/model/metric"
  dataset : exactly as stored in results.json  (e.g. "VRDU Reg.", "W-2")
  model   : "Haiku" or "GPT-4o-mini"
  metric  : "F1", "CBA_s", "delta", or "misbindings"

Run from the repo root (Hindsight Revision/):
    python3 SRC/scripts/gen_table_results.py

Or from SRC/scripts/:
    python3 gen_table_results.py
"""
import json
import os
import sys

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SRC_DIR, "results")
PAPER_DIR   = os.path.join(os.path.dirname(SRC_DIR), "Paper")
TABLES_DIR  = os.path.join(PAPER_DIR, "tables")

DATA_FILE    = os.path.join(RESULTS_DIR, "results.json")
CHANGED_FILE = os.path.join(RESULTS_DIR, "changed_cells.json")
OUTPUT_FILE  = os.path.join(TABLES_DIR, "tab_all_results.tex")
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_cell_key(dataset, model, metric):
    return f"{dataset}/{model}/{metric}"


def makecell(point, ci_lo, ci_hi):
    r"""Plain (non-bold) \makecell with point estimate and CI."""
    return rf"\makecell{{{point} \\ {{\footnotesize [{ci_lo}, {ci_hi}]}}}}"


def makecell_bold(point, ci_lo, ci_hi):
    r"""Bold \makecell for high-misbinding delta cells."""
    return (
        rf"\makecell{{\textbf{{{point}}} \\ "
        rf"{{\footnotesize [\textbf{{{ci_lo}}}, \textbf{{{ci_hi}}}]}}}}"
    )


def wrap_rev(content, changed):
    """Wrap content in \rev{} if this cell is marked changed or new."""
    if changed:
        return rf"\rev{{{content}}}"
    return content


def build_table(rows, changed_set):
    lines = []
    lines.append(r"\begin{tabular}{llccccr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Recall (\%)} "
        r"& \textbf{$\text{CBA}_{\!s}$ (\%)} "
        r"& \textbf{$\Delta$ (\%)} & \textbf{Misbindings} \\"
    )
    lines.append(r"\midrule")

    # Group consecutive rows that share the same dataset
    groups = []
    prev_ds = None
    group = []
    for row in rows:
        if row["dataset"] != prev_ds:
            if group:
                groups.append(group)
            group = [row]
            prev_ds = row["dataset"]
        else:
            group.append(row)
    if group:
        groups.append(group)

    for g_idx, group in enumerate(groups):
        last_group = (g_idx == len(groups) - 1)
        n = len(group)
        for r_idx, row in enumerate(group):
            ds   = row["dataset"]
            mdl  = row["model"]
            bold = row["delta_bold"]

            # ── F1 cell ─────────────────────────────────────────────────────
            f1_content = makecell(row["F1_point"], row["F1_ci_lo"], row["F1_ci_hi"])
            f1_cell = wrap_rev(f1_content,
                               make_cell_key(ds, mdl, "F1") in changed_set)

            # ── CBA_s cell ───────────────────────────────────────────────────
            cba_content = makecell(row["CBA_s_point"],
                                   row["CBA_s_ci_lo"], row["CBA_s_ci_hi"])
            cba_cell = wrap_rev(cba_content,
                                make_cell_key(ds, mdl, "CBA_s") in changed_set)

            # ── delta cell ───────────────────────────────────────────────────
            if bold:
                delta_content = makecell_bold(row["delta_point"],
                                              row["delta_ci_lo"],
                                              row["delta_ci_hi"])
            else:
                delta_content = makecell(row["delta_point"],
                                         row["delta_ci_lo"], row["delta_ci_hi"])
            delta_cell = wrap_rev(delta_content,
                                  make_cell_key(ds, mdl, "delta") in changed_set)

            # ── misbindings cell ─────────────────────────────────────────────
            mb_content = str(row["misbindings"])
            mb_cell = wrap_rev(mb_content,
                               make_cell_key(ds, mdl, "misbindings") in changed_set)

            # ── assemble row ─────────────────────────────────────────────────
            if r_idx == 0:
                first_col = rf"\multirow{{{n}}}{{*}}{{{ds}}}"
            else:
                first_col = ""

            lines.append(
                f"{first_col}\n& {mdl}\n"
                f"& {f1_cell}\n"
                f"& {cba_cell}\n"
                f"& {delta_cell}\n"
                f"& {mb_cell} \\\\"
            )

        # Separator after each group except the last
        if not last_group:
            lines.append(r"\cline{1-6}")
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    data    = load_json(DATA_FILE)
    changed = load_json(CHANGED_FILE)

    rows = data["tab:all_results"]
    changed_list = changed.get("tab:all_results", [])
    # Build a set of cell keys for O(1) lookup
    changed_set = set(changed_list)

    os.makedirs(TABLES_DIR, exist_ok=True)
    table_tex = build_table(rows, changed_set)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(table_tex)
        f.write("\n")

    print(f"Generated: {OUTPUT_FILE}")
    if changed_set:
        print(f"  Changed cells ({len(changed_set)}): {sorted(changed_set)}")
    else:
        print("  No cells marked changed — all cells rendered plain.")


if __name__ == "__main__":
    main()
