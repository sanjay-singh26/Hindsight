"""Patch all CBA notebooks:

1. Rename field_f1 -> field_recall and field_f1_contribution -> field_recall_contribution
   in every CODE cell (output cells are untouched — they will be regenerated on re-run).
2. Fix score_field in CORD and Paystub notebooks: condition cba_strict/cba_soft on
   value_match so that step (v) — concept matched, value wrong — yields CBA=0.

Run from repo root:
    python3 SRC/scripts/patch_notebooks.py
"""
import json, re, os, sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.dirname(SCRIPT_DIR)
NB_DIR      = os.path.join(SRC_DIR, "notebooks")

# ── key-name substitutions applied to ALL notebooks ─────────────────────────
KEY_SUBS = [
    # dict-key strings in source code
    (r'"field_f1_contribution"',  '"field_recall_contribution"'),
    (r"'field_f1_contribution'",  "'field_recall_contribution'"),
    (r'"field_f1"',               '"field_recall"'),
    (r"'field_f1'",               "'field_recall'"),
    # variable names / f-string labels
    (r'\bfield_f1\b(?!_contribution)',  'field_recall'),
    # f-string display labels
    (r'Field F1',                 'Field Recall'),
    (r'field_f1=',                'field_recall='),
]

# ── score_field fix for CORD and Paystub (condition CBA on value_match) ─────
SCORE_FIELD_OLD = '''\
    strict = cba_strict(pred_concept, gt_concept)
    soft = cba_soft(pred_concept, gt_concept, ontology)

    return {
        "value_match": value_match,
        "field_f1_contribution": 1.0 if value_match else 0.0,
        "cba_strict": strict,
        "cba_soft": soft,
        "is_misbinding": value_match and strict == 0.0,'''

SCORE_FIELD_NEW = '''\
    # CBA is only meaningful when value was correctly extracted (protocol step v:
    # concept matched, value wrong → CBA=0, not CBA=1).
    if value_match:
        strict = cba_strict(pred_concept, gt_concept)
        soft = cba_soft(pred_concept, gt_concept, ontology)
    else:
        strict = 0.0
        soft = 0.0

    return {
        "value_match": value_match,
        "field_recall_contribution": 1.0 if value_match else 0.0,
        "cba_strict": strict,
        "cba_soft": soft,
        "is_misbinding": value_match and strict == 0.0,'''

NEEDS_SCORE_FIELD_FIX = {
    "hindsight_cord_cba.ipynb",
    "hindsight_paystub_cba_v2.ipynb",
}


def apply_key_subs(text: str) -> str:
    for pattern, replacement in KEY_SUBS:
        text = re.sub(pattern, replacement, text)
    return text


def fix_score_field(src: str) -> str:
    """Replace the buggy score_field body with the value-conditioned version."""
    # After key subs, field_f1_contribution is already renamed; match new name too
    old_a = SCORE_FIELD_OLD
    old_b = old_a.replace('"field_f1_contribution"', '"field_recall_contribution"')
    for old in (old_a, old_b):
        if old in src:
            src = src.replace(old, SCORE_FIELD_NEW)
    return src


def patch_notebook(path: str, fix_sf: bool) -> tuple[int, int]:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    code_cells_changed = 0
    sf_fixed = 0

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        lines = cell.get("source", [])
        src = "".join(lines)
        new_src = apply_key_subs(src)
        if fix_sf:
            new_src2 = fix_score_field(new_src)
            if new_src2 != new_src:
                sf_fixed += 1
                new_src = new_src2
        if new_src != src:
            code_cells_changed += 1
            # Re-split into lines preserving newlines
            cell["source"] = [
                line + ("\n" if not line.endswith("\n") else "")
                for line in new_src.splitlines()
            ]
            # Strip trailing newline from the last line (notebook convention)
            if cell["source"] and cell["source"][-1].endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return code_cells_changed, sf_fixed


def main():
    notebooks = sorted(f for f in os.listdir(NB_DIR) if f.endswith(".ipynb"))
    total_cells = 0
    total_sf = 0
    for nb_name in notebooks:
        path = os.path.join(NB_DIR, nb_name)
        fix_sf = nb_name in NEEDS_SCORE_FIELD_FIX
        cells, sf = patch_notebook(path, fix_sf)
        total_cells += cells
        total_sf += sf
        tag = " [+score_field fix]" if sf else ""
        print(f"  {nb_name}: {cells} code cell(s) updated{tag}")
    print(f"\nDone. {total_cells} code cells patched, {total_sf} score_field bodies fixed.")


if __name__ == "__main__":
    main()
