"""Category C5: open-weights vision-language model evaluation (Qwen2-VL-7B).

Evaluates the local Qwen2-VL-7B-Instruct (4-bit, via mlx-vlm) on the same
sampling/prompts/ground-truth already used for Haiku/GPT-4o-mini in the
Category C1 re-score, across high-ambiguity (CUAD, W-2) and low-ambiguity
(SROIE, VRDU) domains, per three_category_review.md item C5. Scored with
the same corrected cba_scorer.py used throughout.

Runs entirely locally — no API cost.

Usage:
    python3 scripts/eval_qwen_c5.py --domain cuad --n-samples 50 --out /tmp/qwen_cuad.json
    python3 scripts/eval_qwen_c5.py --domain w2 --n-samples 50 --out /tmp/qwen_w2.json
    python3 scripts/eval_qwen_c5.py --domain sroie --n-samples 50 --out /tmp/qwen_sroie.json
    python3 scripts/eval_qwen_c5.py --domain vrdu_registration --n-samples 50 --out /tmp/qwen_vrdu_reg.json
    python3 scripts/eval_qwen_c5.py --domain vrdu_adbuy --n-samples 50 --out /tmp/qwen_vrdu_adbuy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from local_qwen_client import LocalQwenClient  # noqa: E402
from cba_scorer import score_corpus  # noqa: E402

MODEL_ID = "qwen2-vl-7b-instruct-4bit"


def run_cuad(client, n_samples: int):
    import rescore_cuad as m

    cuad_data = m.ensure_cuad_data()
    span_pool = m.build_span_pool(cuad_data)
    bundles = m.build_bundles(span_pool, n_samples)

    bundle_results = []
    for b in bundles:
        response = client.chat.completions.create(
            model=MODEL_ID,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": m.SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify each clause below:\n\n{b['text']}"},
            ],
        )
        try:
            arr = m.parse_json_array(response.choices[0].message.content)
        except Exception as e:
            print(f"  parse error on bundle {b['bundle_id']}: {e}")
            arr = []
        pred_mapping = m.array_to_pred_mapping(arr)
        bundle_results.append({"bundle_id": b["bundle_id"], "gt_mapping": b["gt_mapping"], "pred_mapping": pred_mapping})
        print(f"  bundle {b['bundle_id']}: {len(pred_mapping)}/15 classified")

    documents = []
    for br in bundle_results:
        gt_items = [(concept, str(cn)) for cn, concept in br["gt_mapping"].items()]
        pred_items = [(concept, str(cn)) for cn, concept in br["pred_mapping"].items()]
        documents.append((gt_items, pred_items))
    return score_corpus(documents), bundle_results


def run_w2(client, n_samples: int):
    import rescore_w2 as m

    samples = m.build_samples(n_samples)
    doc_results = []
    for i, s in enumerate(samples):
        b64 = m.encode_image(s["image_path"])
        response = client.chat.completions.create(
            model=MODEL_ID,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": m.W2_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Extract all fields from this W-2 tax form image."},
                ]},
            ],
        )
        try:
            pred = m.parse_json_object(response.choices[0].message.content)
        except Exception as e:
            print(f"  parse error on {s['doc_id']}: {e}")
            pred = {}
        doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred})
        print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}")

    documents = []
    for d in doc_results:
        gt_items = [(c, v) for c, v in d["ground_truth"].items() if v != "N/A"]
        pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
        documents.append((gt_items, pred_items))
    return score_corpus(documents), doc_results


def run_sroie(client, n_samples: int):
    import rescore_sroie as m

    samples = m.build_samples(n_samples)
    doc_results = []
    for i, s in enumerate(samples):
        b64 = m.encode_image(s["image_path"])
        response = client.chat.completions.create(
            model=MODEL_ID,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": m.SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Extract all fields from this receipt image."},
                ]},
            ],
        )
        try:
            pred = m.parse_json_object(response.choices[0].message.content)
        except Exception as e:
            print(f"  parse error on {s['doc_id']}: {e}")
            pred = {}
        doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred})
        print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}")

    documents = []
    for d in doc_results:
        gt_items = [(c, v) for c, v in d["ground_truth"].items() if v != "N/A"]
        pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
        documents.append((gt_items, pred_items))
    return score_corpus(documents), doc_results


def run_vrdu(client, domain: str, n_samples: int):
    import rescore_vrdu as m

    cfg = m.DOMAIN_CONFIG[domain]
    m.ensure_vrdu_repo()
    samples = m.build_samples(domain, n_samples)
    doc_results = []
    for i, s in enumerate(samples):
        images = m.pdf_to_images(s["pdf_path"])
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{m.encode_image(img)}"}} for img in images]
        content.append({"type": "text", "text": cfg["extract_instruction"]})
        response = client.chat.completions.create(
            model=MODEL_ID,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": content},
            ],
        )
        try:
            pred = m.parse_json_object(response.choices[0].message.content)
        except Exception as e:
            print(f"  parse error on {s['doc_id']}: {e}")
            pred = {}
        doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred})
        print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}")

    documents = []
    for d in doc_results:
        gt_items = [(c, v) for c, v in d["ground_truth"].items()]
        pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
        documents.append((gt_items, pred_items))
    return score_corpus(documents), doc_results


RUNNERS = {
    "cuad": lambda client, n: run_cuad(client, n),
    "w2": lambda client, n: run_w2(client, n),
    "sroie": lambda client, n: run_sroie(client, n),
    "vrdu_registration": lambda client, n: run_vrdu(client, "registration", n),
    "vrdu_adbuy": lambda client, n: run_vrdu(client, "adbuy", n),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(RUNNERS.keys()))
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--out", default="/tmp/qwen_eval.json")
    args = parser.parse_args()

    print(f"Loading local Qwen2-VL-7B model...")
    t0 = time.time()
    client = LocalQwenClient()
    print(f"Loaded in {time.time() - t0:.1f}s")

    print(f"\n=== Domain: {args.domain} ({args.n_samples} samples) ===")
    corpus, raw_results = RUNNERS[args.domain](client, args.n_samples)

    print(f"\nDONE: Rec_val={corpus.rec_val:.4f} CBA_cond={corpus.cba_cond:.4f} "
          f"ACC_joint={corpus.acc_joint:.4f} Delta={corpus.delta:.4f} misbindings={corpus.n_misbindings}")

    with open(args.out, "w") as f:
        json.dump({"domain": args.domain, "model_id": MODEL_ID, "scorer_result": corpus.to_dict(), "raw_results": raw_results}, f, indent=2)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
