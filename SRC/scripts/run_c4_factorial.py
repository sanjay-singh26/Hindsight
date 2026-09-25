"""Category C4: run the controlled density x semantic-overlap factorial study on W-2.

For each of the 20 (overlap, density) cells defined in c4_factorial_design.py,
restricts ground truth to the cell's N active concepts, builds a
schema-constrained extraction prompt listing only those N concepts, and
runs both models on the same 50 W-2 forms used throughout Category C1/C3.
Scores each cell with cba_scorer.py. Saves incrementally (one file per
cell) so a long run can be resumed/inspected mid-flight, plus a combined
summary file at the end.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/run_c4_factorial.py --n-samples 50 --out-dir /tmp/c4_cells
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rescore_w2 as base  # noqa: E402
from c4_factorial_design import OVERLAP_ORDERINGS, DENSITY_LEVELS, active_fields  # noqa: E402
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"

CONCEPT_DESCRIPTIONS = {
    "employee_ssn": "Employee Social Security Number",
    "employer_ein": "Employer Identification Number",
    "employer_name": "Employer/company name",
    "employee_name": "Employee full name",
    "control_number": "Control number",
    "employer_address": "Employer street address",
    "employer_city_state_zip": "Employer city, state, and ZIP",
    "employee_address": "Employee street address",
    "employee_city_state_zip": "Employee city, state, and ZIP",
    "wages_tips_compensation": "Total wages, tips, and other compensation",
    "social_security_wages": "Wages subject to Social Security tax",
    "medicare_wages": "Wages and tips subject to Medicare tax",
    "social_security_tips": "Tips subject to Social Security tax",
    "allocated_tips": "Allocated tips",
    "federal_tax_withheld": "Federal income tax withheld",
    "social_security_tax": "Social Security tax withheld",
    "medicare_tax": "Medicare tax withheld",
    "dependent_care_benefits": "Dependent care benefits",
    "nonqualified_plans": "Nonqualified deferred compensation plans",
    "box_12a_code": "First deferred compensation/benefit code",
    "box_12a_value": "First deferred compensation/benefit amount",
    "box_12b_code": "Second deferred compensation/benefit code",
    "box_12b_value": "Second deferred compensation/benefit amount",
    "box_12c_code": "Third deferred compensation/benefit code",
    "box_12c_value": "Third deferred compensation/benefit amount",
    "box_12d_code": "Fourth deferred compensation/benefit code",
    "box_12d_value": "Fourth deferred compensation/benefit amount",
    "statutory_employee": "Is the employee a statutory employee (Yes/No)",
    "retirement_plan": "Is the employee in a retirement plan (Yes/No)",
    "third_party_sick_pay": "Was third-party sick pay provided (Yes/No)",
    "state_1": "Primary state abbreviation",
    "state_1_employer_id": "Employer state ID (primary state)",
    "state_1_wages": "State wages for primary state",
    "state_1_income_tax": "State income tax withheld for primary state",
    "local_1_wages": "Local wages for primary locality",
    "local_1_income_tax": "Local income tax withheld for primary locality",
    "local_1_name": "Primary locality name",
    "state_2": "Secondary state abbreviation",
    "state_2_employer_id": "Employer state ID (secondary state)",
    "state_2_wages": "State wages for secondary state",
    "state_2_income_tax": "State income tax withheld for secondary state",
    "local_2_wages": "Local wages for secondary locality",
    "local_2_income_tax": "Local income tax withheld for secondary locality",
    "local_2_name": "Secondary locality name",
}


def build_prompt(active: list) -> str:
    lines = [
        "You are a document AI extraction system. You extract structured data from W-2 tax form images.",
        "",
        "Given an image of a W-2 form, extract values for the following canonical concept keys.",
        "You MUST use exactly these keys — do NOT use the box numbers or labels from the form.",
        "",
        "CANONICAL CONCEPT KEYS:",
    ]
    for c in active:
        lines.append(f"- {c}: {CONCEPT_DESCRIPTIONS[c]}")
    lines += [
        "",
        "RULES:",
        f"1. Return ONLY a JSON object with exactly these {len(active)} keys.",
        "2. Monetary values: numeric string WITHOUT $ or commas (e.g., \"4523.67\").",
        "3. SSN format: XXX-XX-XXXX. EIN format: XX-XXXXXXX.",
        "4. Boolean fields (checkboxes): \"Yes\" or \"No\".",
        "5. If a field is not present or empty on the form, use \"N/A\".",
        "6. Return ONLY valid JSON, no other text.",
    ]
    return "\n".join(lines)


def parse_json_object(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def extract(client, model_id: str, image_path: str, prompt: str, max_retries=5):
    b64 = base.encode_image(image_path)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": "Extract all fields from this W-2 tax form image."},
                    ]},
                ],
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            return parse_json_object(content), cost
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt < max_retries - 1:
                wait = min(10 * (2 ** attempt), 60) if is_rate_limit else 2 ** (attempt + 1)
                print(f"      retry {attempt + 1}/{max_retries - 1} in {wait}s: {type(e).__name__}: {e}")
                time.sleep(wait)
            else:
                print(f"      GIVING UP after {max_retries} attempts: {e}")
                return {}, 0.0


def get_balance(api_key):
    import httpx
    try:
        r = httpx.get(f"{GATEWAY_BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--out-dir", default="/tmp/c4_cells")
    parser.add_argument("--models", nargs="+", default=["haiku", "gpt4o-mini"])
    parser.add_argument("--overlaps", nargs="+", default=list(OVERLAP_ORDERINGS.keys()))
    parser.add_argument("--densities", nargs="+", type=int, default=DENSITY_LEVELS)
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    os.makedirs(args.out_dir, exist_ok=True)

    balance_start = get_balance(api_key)
    print(f"Gateway balance at start: ${balance_start}")

    samples = base.build_samples(args.n_samples)
    print(f"Sampled {len(samples)} W-2 forms (shared across all cells)")

    summary = {}
    total_cells = len(args.overlaps) * len(args.densities)
    cell_i = 0

    for overlap in args.overlaps:
        for density in args.densities:
            cell_i += 1
            cell_key = f"{overlap}_N{density}"
            out_path = os.path.join(args.out_dir, f"{cell_key}.json")
            if os.path.exists(out_path):
                print(f"[{cell_i}/{total_cells}] {cell_key}: already done, skipping")
                summary[cell_key] = json.load(open(out_path))["summary"]
                continue

            active = active_fields(overlap, density)
            prompt = build_prompt(active)
            print(f"\n[{cell_i}/{total_cells}] === {cell_key} (fields: {active[:3]}{'...' if density > 3 else ''}) ===")

            cell_results = {}
            for model_name in args.models:
                model_id = base.MODELS[model_name]
                doc_results = []
                for i, s in enumerate(samples):
                    gt_subset = {c: s["ground_truth"].get(c, "N/A") for c in active}
                    pred, cost = extract(client, model_id, s["image_path"], prompt)
                    doc_results.append({"doc_id": s["doc_id"], "ground_truth": gt_subset, "predicted": pred, "cost": cost})
                    time.sleep(0.3)

                documents = []
                for d in doc_results:
                    gt_items = [(c, v) for c, v in d["ground_truth"].items() if v != "N/A"]
                    pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
                    documents.append((gt_items, pred_items))

                corpus = score_corpus(documents)
                cell_results[model_name] = {"model_id": model_id, "scorer_result": corpus.to_dict(), "documents": doc_results}
                print(f"  {model_name:12s}: Rec_val={corpus.rec_val:.3f} CBA_cond={corpus.cba_cond:.3f} "
                      f"Delta={corpus.delta:.3f} misbindings={corpus.n_misbindings}")

            with open(out_path, "w") as f:
                json.dump({"overlap": overlap, "density": density, "active_fields": active, "cell": cell_results,
                           "summary": {m: cell_results[m]["scorer_result"] for m in cell_results}}, f, indent=2)

            summary[cell_key] = {m: cell_results[m]["scorer_result"] for m in cell_results}
            balance_now = get_balance(api_key)
            print(f"  saved to {out_path}. Balance: ${balance_now}")

    with open(os.path.join(args.out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    balance_end = get_balance(api_key)
    print(f"\nAll cells complete. Balance at end: ${balance_end}")
    if balance_start is not None and balance_end is not None:
        print(f"Total spend: ${balance_start - balance_end:.4f}")


if __name__ == "__main__":
    main()
