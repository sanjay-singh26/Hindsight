"""Category C1 re-scoring: W-2, via LLMGateway (vision).

Reproduces notebooks/hindsight_w2_cba.ipynb's sampling (same
random.seed(42), same HuggingFace source, same 44-concept schema),
extracts via vision through LLMGateway, and scores with the corrected
cba_scorer.py.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_w2.py --n-samples 3 --out /tmp/w2_pilot.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"
IMG_DIR = "/tmp/w2_images"

W2_ONTOLOGY_FAMILIES = {
    "identity": ["employee_ssn", "employer_ein", "employer_name", "employee_name", "control_number"],
    "address": ["employer_address", "employer_city_state_zip", "employee_address", "employee_city_state_zip"],
    "compensation": ["wages_tips_compensation", "social_security_wages", "medicare_wages", "social_security_tips", "allocated_tips"],
    "withholding": ["federal_tax_withheld", "social_security_tax", "medicare_tax"],
    "benefits": ["dependent_care_benefits", "nonqualified_plans"],
    "box12": ["box_12a_code", "box_12a_value", "box_12b_code", "box_12b_value",
              "box_12c_code", "box_12c_value", "box_12d_code", "box_12d_value"],
    "box13": ["statutory_employee", "retirement_plan", "third_party_sick_pay"],
    "state_local": ["state_1", "state_1_employer_id", "state_1_wages", "state_1_income_tax",
                     "local_1_wages", "local_1_income_tax", "local_1_name",
                     "state_2", "state_2_employer_id", "state_2_wages", "state_2_income_tax",
                     "local_2_wages", "local_2_income_tax", "local_2_name"],
}
ALL_CONCEPTS = [c for fam in W2_ONTOLOGY_FAMILIES.values() for c in fam]

W2_FIELD_MAP = {
    "box_a_employee_ssn": "employee_ssn",
    "box_b_employer_identification_number": "employer_ein",
    "box_c_employer_name": "employer_name",
    "box_e_employee_name": "employee_name",
    "box_d_control_number": "control_number",
    "box_c_employer_street_address": "employer_address",
    "box_c_employer_city_state_zip": "employer_city_state_zip",
    "box_e_employee_street_address": "employee_address",
    "box_e_employee_city_state_zip": "employee_city_state_zip",
    "box_1_wages": "wages_tips_compensation",
    "box_3_social_security_wages": "social_security_wages",
    "box_5_medicare_wages": "medicare_wages",
    "box_7_social_security_tips": "social_security_tips",
    "box_8_allocated_tips": "allocated_tips",
    "box_2_federal_tax_withheld": "federal_tax_withheld",
    "box_4_social_security_tax_withheld": "social_security_tax",
    "box_6_medicare_wages_tax_withheld": "medicare_tax",
    "box_10_dependent_care_benefits": "dependent_care_benefits",
    "box_11_nonqualified_plans": "nonqualified_plans",
    "box_12a_code": "box_12a_code", "box_12a_value": "box_12a_value",
    "box_12b_code": "box_12b_code", "box_12b_value": "box_12b_value",
    "box_12c_code": "box_12c_code", "box_12c_value": "box_12c_value",
    "box_12d_code": "box_12d_code", "box_12d_value": "box_12d_value",
    "box_13_statutory_employee": "statutory_employee",
    "box_13_retirement_plan": "retirement_plan",
    "box_13_third_party_sick_pay": "third_party_sick_pay",
    "box_15_1_state": "state_1",
    "box_15_1_employer_state_id": "state_1_employer_id",
    "box_16_1_state_wages": "state_1_wages",
    "box_17_1_state_income_tax": "state_1_income_tax",
    "box_18_1_local_wages": "local_1_wages",
    "box_19_1_local_income_tax": "local_1_income_tax",
    "box_20_1_locality_name": "local_1_name",
    "box_15_2_state": "state_2",
    "box_15_2_employer_state_id": "state_2_employer_id",
    "box_16_2_state_wages": "state_2_wages",
    "box_17_2_state_income_tax": "state_2_income_tax",
    "box_18_2_local_wages": "local_2_wages",
    "box_19_2_local_income_tax": "local_2_income_tax",
    "box_20_2_locality_name": "local_2_name",
}

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}

W2_SYSTEM_PROMPT = """You are a document AI extraction system. You extract structured data from W-2 tax form images.

Given an image of a W-2 form, extract values for the following canonical concept keys.
You MUST use exactly these keys — do NOT use the box numbers or labels from the form.

CANONICAL CONCEPT KEYS:
- employee_ssn: Employee Social Security Number
- employer_ein: Employer Identification Number
- employer_name: Employer/company name
- employee_name: Employee full name
- control_number: Control number
- employer_address: Employer street address
- employer_city_state_zip: Employer city, state, and ZIP
- employee_address: Employee street address
- employee_city_state_zip: Employee city, state, and ZIP
- wages_tips_compensation: Total wages, tips, and other compensation
- social_security_wages: Wages subject to Social Security tax
- medicare_wages: Wages and tips subject to Medicare tax
- social_security_tips: Tips subject to Social Security tax
- allocated_tips: Allocated tips
- federal_tax_withheld: Federal income tax withheld
- social_security_tax: Social Security tax withheld
- medicare_tax: Medicare tax withheld
- dependent_care_benefits: Dependent care benefits
- nonqualified_plans: Nonqualified deferred compensation plans
- box_12a_code: First deferred compensation/benefit code
- box_12a_value: First deferred compensation/benefit amount
- box_12b_code: Second deferred compensation/benefit code
- box_12b_value: Second deferred compensation/benefit amount
- box_12c_code: Third deferred compensation/benefit code
- box_12c_value: Third deferred compensation/benefit amount
- box_12d_code: Fourth deferred compensation/benefit code
- box_12d_value: Fourth deferred compensation/benefit amount
- statutory_employee: Is the employee a statutory employee (Yes/No)
- retirement_plan: Is the employee in a retirement plan (Yes/No)
- third_party_sick_pay: Was third-party sick pay provided (Yes/No)
- state_1: Primary state abbreviation
- state_1_employer_id: Employer state ID (primary state)
- state_1_wages: State wages for primary state
- state_1_income_tax: State income tax withheld for primary state
- local_1_wages: Local wages for primary locality
- local_1_income_tax: Local income tax withheld for primary locality
- local_1_name: Primary locality name
- state_2: Secondary state abbreviation
- state_2_employer_id: Employer state ID (secondary state)
- state_2_wages: State wages for secondary state
- state_2_income_tax: State income tax withheld for secondary state
- local_2_wages: Local wages for secondary locality
- local_2_income_tax: Local income tax withheld for secondary locality
- local_2_name: Secondary locality name

RULES:
1. Return ONLY a JSON object with exactly these 44 keys.
2. Monetary values: numeric string WITHOUT $ or commas (e.g., "4523.67").
3. SSN format: XXX-XX-XXXX. EIN format: XX-XXXXXXX.
4. Boolean fields (checkboxes): "Yes" or "No".
5. If a field is not present or empty on the form, use "N/A".
6. Return ONLY valid JSON, no other text."""


def convert_gt(gt_parse: dict) -> dict:
    canonical = {}
    for hf_key, concept_id in W2_FIELD_MAP.items():
        val = gt_parse.get(hf_key)
        if val is None or val == "":
            canonical[concept_id] = "N/A"
        elif isinstance(val, bool):
            canonical[concept_id] = "Yes" if val else "No"
        elif isinstance(val, float):
            canonical[concept_id] = f"{val:.2f}"
        else:
            canonical[concept_id] = str(val)
    return canonical


def build_samples(n_samples: int) -> list:
    from datasets import load_dataset
    from PIL import Image as PILImage

    ds = load_dataset("singhsays/fake-w2-us-tax-form-dataset")
    train_data = ds["train"]
    print(f"W-2 dataset loaded: {len(train_data)} forms")

    random.seed(42)
    os.makedirs(IMG_DIR, exist_ok=True)
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    selected = sorted(indices[:50])[:n_samples]

    samples = []
    for i, idx in enumerate(selected):
        rec = train_data[idx]
        doc_id = f"w2_{i:04d}"
        gt_raw = json.loads(rec["ground_truth"]) if isinstance(rec["ground_truth"], str) else rec["ground_truth"]
        gt_parse = gt_raw.get("gt_parse", gt_raw)
        gt = convert_gt(gt_parse)

        img_path = os.path.join(IMG_DIR, f"{doc_id}.png")
        img = rec["image"]
        if isinstance(img, PILImage.Image):
            img.save(img_path)
        elif isinstance(img, bytes):
            with open(img_path, "wb") as f:
                f.write(img)

        samples.append({"doc_id": doc_id, "image_path": img_path, "ground_truth": gt})
    return samples


def encode_image(image_path: str, max_width: int = 1024) -> str:
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def parse_json_object(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def extract(client, model_id: str, image_path: str, max_retries=5):
    b64 = encode_image(image_path)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": W2_SYSTEM_PROMPT},
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
                wait = min(10 * (2 ** attempt), 120) if is_rate_limit else 2 ** (attempt + 1)
                print(f"    retry {attempt + 1}/{max_retries - 1} in {wait}s: {type(e).__name__}: {e}")
                time.sleep(wait)
            else:
                raise


def get_balance(api_key) -> float | None:
    import httpx
    try:
        r = httpx.get(f"{GATEWAY_BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--out", default="/tmp/w2_pilot.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    balance_before = get_balance(api_key)
    print(f"Gateway balance before run: ${balance_before}")

    samples = build_samples(args.n_samples)
    print(f"Sampled {len(samples)} W-2 forms")

    results = {}
    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        doc_results = []
        for i, s in enumerate(samples):
            pred, cost = extract(client, model_id, s["image_path"])
            doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred, "cost": cost})
            print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}: cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, v) for c, v in d["ground_truth"].items() if v != "N/A"]
            pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
            documents.append((gt_items, pred_items))

        corpus = score_corpus(documents)
        results[model_name] = {"model_id": model_id, "scorer_result": corpus.to_dict(), "documents": doc_results}
        print(f"  DONE: Rec_val={corpus.rec_val:.4f} CBA_cond={corpus.cba_cond:.4f} "
              f"ACC_joint={corpus.acc_joint:.4f} Delta={corpus.delta:.4f} misbindings={corpus.n_misbindings}")

    balance_after = get_balance(api_key)
    print(f"\nGateway balance after run: ${balance_after}")
    if balance_before is not None and balance_after is not None:
        print(f"Actual spend this run: ${balance_before - balance_after:.6f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
