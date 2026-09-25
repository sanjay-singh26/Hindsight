"""Category C1 re-scoring: VRDU Registration (FARA) and VRDU Ad-buy, via LLMGateway (vision, multi-page PDF).

Reproduces both notebooks' sampling (same random.seed(42), same VRDU
GitHub dataset, same PDF->image rendering via PyMuPDF), extracts via
vision through LLMGateway, and scores with the corrected cba_scorer.py.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_vrdu.py --domain registration --n-samples 3 --out /tmp/vrdu_reg_pilot.json
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_vrdu.py --domain adbuy --n-samples 3 --out /tmp/vrdu_adbuy_pilot.json
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"
VRDU_BASE = "/tmp/vrdu"
MAX_PAGES = 3
RENDER_DPI = 200

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}

DOMAIN_CONFIG = {
    "registration": {
        "subdir": "registration-form",
        "expected_pdfs": 1900,
        "min_concepts": 3,
        "eval_concepts": [
            "registration_num", "registrant_name", "foreign_principle_name",
            "signer_name", "signer_title", "file_date",
        ],
        "doc_prefix": "fara",
        "system_prompt": """You are a document AI extraction system. You extract structured data from FARA (Foreign Agents Registration Act) registration form images.

Given an image (or multiple pages) of a FARA registration/amendment form, extract values for the following canonical concept keys.

CANONICAL CONCEPT KEYS:
- registration_num: The FARA registration number (numeric, usually near the top of the form)
- registrant_name: Name of the registering agent — the US-based law firm, lobbying organization, or individual doing the representation
- foreign_principle_name: Name of the foreign government, organization, company, or entity being represented
- signer_name: Name of the person who signed the form (found in the execution/signature section)
- signer_title: Title or position of the person who signed (e.g., "Partner", "Managing Director")
- file_date: Date the form was filed or signed (found near the signature)

IMPORTANT DISAMBIGUATION RULES:
- registrant_name is the AGENT (US-based entity doing lobbying/representation work)
- foreign_principle_name is the FOREIGN entity being represented (a government, foreign company, or foreign organization)
- signer_name is the specific PERSON who signed, who may be a partner/employee of the registrant
- file_date is the date of signing/filing, NOT other dates mentioned in the document body

RULES:
1. Return ONLY a JSON object with exactly these 6 keys.
2. If a field is not present or you cannot determine it, use "N/A".
3. Return ONLY valid JSON, no other text.""",
        "extract_instruction": "Extract all document-level fields from this FARA registration form.",
    },
    "adbuy": {
        "subdir": "ad-buy-form",
        "expected_pdfs": 600,
        "min_concepts": 5,
        "eval_concepts": [
            "advertiser", "agency", "contract_num", "flight_from", "flight_to",
            "gross_amount", "product", "tv_address", "property",
        ],
        "doc_prefix": "adbuy",
        "system_prompt": """You are a document AI extraction system. You extract structured data from political advertising disclosure form images (FCC filings).

Given an image (or multiple pages) of an ad-buy form/invoice, extract values for the following canonical concept keys.

CANONICAL CONCEPT KEYS:
- advertiser: Name of the political committee, campaign, or organization buying the ads
- agency: Name of the media buying agency (intermediary firm)
- contract_num: Contract number, order number, or invoice number
- flight_from: Start date of the overall advertising campaign/flight period
- flight_to: End date of the overall advertising campaign/flight period
- gross_amount: Total gross amount for the entire order/invoice
- product: Product or campaign name (may be similar to advertiser name)
- tv_address: Mailing/remit address of the TV station
- property: TV station call sign (e.g., KMSP, WJLA, WTTG)

IMPORTANT DISAMBIGUATION RULES:
- flight_from/flight_to are the OVERALL campaign dates (often labeled "Order Flight" or "Flight Dates"), NOT individual program/spot dates
- gross_amount is the TOTAL amount for the entire invoice, NOT individual line item amounts
- advertiser is the buyer/committee name, agency is the intermediary firm that placed the buy, product is the campaign label
- contract_num is the primary order/contract/invoice number
- property is the station call sign (e.g., "KMSP"), NOT the station name or channel number

RULES:
1. Return ONLY a JSON object with exactly these 9 keys.
2. If a field is not present or you cannot determine it, use "N/A".
3. Dates: use the format shown on the form (e.g., "12/30/19" or "03/29/20").
4. Dollar amounts: include $ and commas as shown (e.g., "$5,625.00").
5. Return ONLY valid JSON, no other text.""",
        "extract_instruction": "Extract all document-level fields from this ad-buy form.",
    },
}


def ensure_vrdu_repo():
    # Both domains share one shallow clone of the VRDU repo.
    marker = os.path.join(VRDU_BASE, "registration-form", "main", "pdfs")
    if os.path.isdir(marker) and len(os.listdir(marker)) > 1900:
        print("VRDU repo already present.")
        return
    if os.path.exists(VRDU_BASE):
        shutil.rmtree(VRDU_BASE)
    print("Cloning google-research-datasets/vrdu (shallow, ~600MB total)...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/google-research-datasets/vrdu.git", VRDU_BASE],
        check=True,
    )
    print("Clone complete.")


def load_docs(domain: str) -> list:
    cfg = DOMAIN_CONFIG[domain]
    vrdu_path = os.path.join(VRDU_BASE, cfg["subdir"], "main")
    pdf_dir = os.path.join(vrdu_path, "pdfs")
    n_pdfs = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
    print(f"{domain}: {n_pdfs} PDFs found")

    dataset_path = os.path.join(vrdu_path, "dataset.jsonl")
    if not os.path.exists(dataset_path):
        gz_path = dataset_path + ".gz"
        with gzip.open(gz_path, "rb") as f_in, open(dataset_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    eval_concepts = cfg["eval_concepts"]
    all_docs = []
    with open(dataset_path) as f:
        for line in f:
            doc = json.loads(line)
            gt = {}
            for ann in doc["annotations"]:
                entity_name = ann[0]
                if isinstance(entity_name, str) and entity_name in eval_concepts:
                    values = ann[1]
                    if values:
                        text = values[0][0].strip()
                        if text and entity_name not in gt:
                            gt[entity_name] = text
            pdf_path = os.path.join(pdf_dir, doc["filename"])
            if len(gt) >= cfg["min_concepts"] and os.path.exists(pdf_path):
                all_docs.append({"filename": doc["filename"], "pdf_path": pdf_path, "ground_truth": gt})

    print(f"{domain}: {len(all_docs)} docs pass concept+PDF filters")
    return all_docs


def build_samples(domain: str, n_samples: int) -> list:
    cfg = DOMAIN_CONFIG[domain]
    all_docs = load_docs(domain)
    random.seed(42)
    indices = list(range(len(all_docs)))
    random.shuffle(indices)
    selected = sorted(indices[:50])[:n_samples]

    samples = []
    for i, idx in enumerate(selected):
        doc = all_docs[idx]
        samples.append({
            "doc_id": f"{cfg['doc_prefix']}_{i:04d}",
            "pdf_path": doc["pdf_path"],
            "ground_truth": doc["ground_truth"],
        })
    return samples


def pdf_to_images(pdf_path: str, max_pages: int = MAX_PAGES, dpi: int = RENDER_DPI) -> list:
    import fitz
    from PIL import Image as PILImage

    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def encode_image(img, max_width: int = 1024) -> str:
    from PIL import Image as PILImage

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


def extract(client, model_id: str, pdf_path: str, system_prompt: str, instruction: str, max_retries=5):
    images = pdf_to_images(pdf_path)
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}} for img in images]
    content.append({"type": "text", "text": instruction})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            )
            raw_content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            return parse_json_object(raw_content), cost
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
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_CONFIG.keys()))
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--out", default="/tmp/vrdu_pilot.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    cfg = DOMAIN_CONFIG[args.domain]
    balance_before = get_balance(api_key)
    print(f"Gateway balance before run: ${balance_before}")

    ensure_vrdu_repo()
    samples = build_samples(args.domain, args.n_samples)
    print(f"Sampled {len(samples)} {args.domain} documents")

    results = {}
    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        doc_results = []
        for i, s in enumerate(samples):
            pred, cost = extract(client, model_id, s["pdf_path"], cfg["system_prompt"], cfg["extract_instruction"])
            doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred, "cost": cost})
            print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}: cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, v) for c, v in d["ground_truth"].items()]
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
