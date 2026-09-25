"""Category C1 re-scoring: LEDGAR, via LLMGateway.

Reproduces notebooks/hindsight_ledgar_cba.ipynb's bundle construction
(same EVAL_CONCEPTS, same provision-pool logic, same random.seed(42)),
classifies each bundle with both models through LLMGateway, and scores
with the corrected cba_scorer.py.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_ledgar.py --n-bundles 50 --out /tmp/ledgar_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"

EVAL_CONCEPTS = [
    "Death", "Disability", "Terminations",
    "Base Salary", "Benefits",
    "Tax Withholdings", "Taxes", "Withholdings",
    "Indemnifications", "Indemnity",
    "Vesting", "Forfeitures",
    "Employment", "Duties", "Positions",
]

MAX_PROVISION_LENGTH = 1500

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}

SYSTEM_PROMPT = """You are a legal document classification system. Given a set of numbered legal provisions from a contract, classify each provision into exactly one of the following categories.

Categories:
- Death: Provisions triggered by death of employee/participant
- Disability: Provisions triggered by disability/incapacity of employee
- Terminations: General termination provisions (not death or disability specific)
- Base Salary: Provisions defining base salary or base compensation amount
- Benefits: Provisions about employee benefits (health, retirement, PTO, perquisites)
- Tax Withholdings: Provisions about withholding taxes from compensation or awards
- Taxes: Provisions about tax obligations, tax treatment, or tax gross-up
- Withholdings: Provisions about deducting/withholding amounts from compensation
- Indemnifications: Provisions granting indemnification rights or protections
- Indemnity: Provisions imposing indemnity obligations (hold-harmless)
- Vesting: Provisions about equity/award vesting schedules or conditions
- Forfeitures: Provisions about equity/award forfeiture upon certain events
- Employment: Provisions about the employment relationship and engagement terms
- Duties: Provisions about job duties, responsibilities, and reporting
- Positions: Provisions about job title, position, and role description

Return a JSON array where each item has:
- "clause": the provision number (integer)
- "concept": the category name from the list above (exact string match)

Rules:
- Classify EVERY provision — do not skip any
- Each provision gets exactly one category
- Use the exact category names listed above"""


def build_provision_pool() -> dict:
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["test"].features["label"].names

    provision_pool = defaultdict(list)
    for split in ["train", "test", "validation"]:
        for i in range(len(ds[split])):
            ex = ds[split][i]
            concept = label_names[ex["label"]]
            if concept in EVAL_CONCEPTS:
                text = ex["text"].strip()
                if len(text) > MAX_PROVISION_LENGTH:
                    text = text[:MAX_PROVISION_LENGTH] + "..."
                provision_pool[concept].append(text)
    return provision_pool


def build_bundles(provision_pool: dict, n_bundles: int) -> list:
    random.seed(42)
    shuffled_pools = {}
    for concept in EVAL_CONCEPTS:
        pool = provision_pool[concept].copy()
        random.shuffle(pool)
        shuffled_pools[concept] = pool

    pool_idx = {c: 0 for c in EVAL_CONCEPTS}
    bundles = []
    for bundle_i in range(n_bundles):
        provisions = []
        for concept in EVAL_CONCEPTS:
            idx = pool_idx[concept] % len(shuffled_pools[concept])
            text = shuffled_pools[concept][idx]
            pool_idx[concept] += 1
            provisions.append((concept, text))
        random.shuffle(provisions)

        doc_parts = []
        gt_mapping = {}
        for clause_num, (concept, text) in enumerate(provisions, 1):
            doc_parts.append(f"Provision {clause_num}:\n{text}")
            gt_mapping[clause_num] = concept

        bundles.append({
            "bundle_id": bundle_i,
            "text": "\n\n".join(doc_parts),
            "gt_mapping": gt_mapping,
        })
    return bundles


def parse_json_array(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def classify_bundle(client, model_id: str, bundle_text: str, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify each provision below:\n\n{bundle_text}"},
                ],
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            return parse_json_array(content), cost
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) if is_rate_limit else base_delay
                print(f"    retry {attempt + 1}/{max_retries - 1} in {delay}s: {type(e).__name__}: {e}")
                time.sleep(delay)
            else:
                raise


def array_to_pred_mapping(arr) -> dict:
    pred = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        clause = item.get("clause") or item.get("clause_number") or item.get("provision")
        concept = item.get("concept", "")
        if clause is not None and concept:
            pred[int(clause)] = concept
    return pred


def get_balance(api_key) -> float | None:
    import httpx
    try:
        r = httpx.get(
            f"{GATEWAY_BASE_URL}/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bundles", type=int, default=3)
    parser.add_argument("--out", default="/tmp/ledgar_pilot.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    balance_before = get_balance(api_key)
    print(f"Gateway balance before run: ${balance_before}")

    print("Loading LEDGAR dataset...")
    provision_pool = build_provision_pool()
    print("Provision pool sizes:")
    for c in EVAL_CONCEPTS:
        print(f"  {c:<25} {len(provision_pool[c]):>5}")

    bundles = build_bundles(provision_pool, args.n_bundles)
    print(f"Built {len(bundles)} bundles ({len(bundles) * 15} classifications per model)")

    results = {}
    total_cost = 0.0

    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        bundle_results = []
        for b in bundles:
            arr, cost = classify_bundle(client, model_id, b["text"])
            pred_mapping = array_to_pred_mapping(arr)
            bundle_results.append({
                "bundle_id": b["bundle_id"],
                "gt_mapping": b["gt_mapping"],
                "pred_mapping": pred_mapping,
                "raw_array": arr,
                "cost": cost,
            })
            if cost:
                total_cost += cost
            if (b["bundle_id"] + 1) % 10 == 0 or b["bundle_id"] == 0:
                print(f"  bundle {b['bundle_id']}: {len(pred_mapping)}/15 classified, cost=${cost}")
            time.sleep(1.0)

        documents = []
        for br in bundle_results:
            gt_items = [(concept, str(cn)) for cn, concept in br["gt_mapping"].items()]
            pred_items = [(concept, str(cn)) for cn, concept in br["pred_mapping"].items()]
            documents.append((gt_items, pred_items))

        corpus = score_corpus(documents)
        results[model_name] = {
            "model_id": model_id,
            "scorer_result": corpus.to_dict(),
            "bundles": bundle_results,
        }
        print(f"  DONE: Rec_val={corpus.rec_val:.4f} CBA_cond={corpus.cba_cond:.4f} "
              f"ACC_joint={corpus.acc_joint:.4f} Delta={corpus.delta:.4f} "
              f"misbindings={corpus.n_misbindings}")

    balance_after = get_balance(api_key)
    print(f"\nGateway balance after run: ${balance_after}")
    if balance_before is not None and balance_after is not None:
        print(f"Actual spend this run (from balance delta): ${balance_before - balance_after:.6f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
