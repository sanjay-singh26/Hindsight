# Hindsight CBA Experiments: Cross-Domain Results

This document summarizes the Concept-Binding Accuracy (CBA) experiments conducted across six document domains, testing whether LLMs correctly bind extracted values to semantic concepts.

## Key Definitions

| Metric | Definition |
|--------|------------|
| **Field F1** | Extraction coverage — did the model produce a value for each field? |
| **CBA-strict** | Exact concept match — was the value assigned to the correct concept? |
| **CBA-soft** | Partial credit — same family = 0.5, same concept = 1.0, different family = 0.0 |
| **Delta** | `Field F1 - CBA-strict`. Positive = misbindings (values correct, concepts wrong) |
| **Misbinding** | A case where the model extracts the correct value but binds it to the wrong concept |

## Models Tested

| Model | Provider | Role |
|-------|----------|------|
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Anthropic | Smaller frontier model |
| GPT-4o-mini | OpenAI | Smaller frontier model |

## Cross-Domain Summary

| Domain | Dataset | Documents | Concepts | Total Misbindings | Haiku Delta | GPT-4o-mini Delta | Dominant Pattern |
|--------|---------|-----------|----------|:-----------------:|:-----------:|:-----------------:|------------------|
| **Paystubs** | Synthetic | 50 | 14 | 6 | -0.152 | -0.205 | Extraction-dominated (Tesseract OCR bottleneck); 6 total misbindings, all within-family |
| **Receipts (SROIE)** | SROIE | 50 | 4 GT | 3 | 0.000 | +0.015 | Near-zero misbinding |
| **Receipts (CORD)** | CORD v2 | 93 | 13 | 53 | -0.019 (vision) / -0.099 (OCR) | -0.003 (vision) / -0.160 (OCR) | Extraction errors dominate; low misbinding domain |
| **W-2 Tax Forms** | Synthetic W-2 | 50 | 44 | **421** | **+0.128** | **+0.094** | High concept density → misbinding |
| **Employment Law** | LEDGAR | 50 bundles | 15 | **350** | **+0.207** | **+0.260** | Semantic overlap → misbinding |
| **Commercial Law** | CUAD v1 | 50 bundles | 15 | **434** | **+0.244** | **+0.335** | Multi-label clauses → misbinding (strongest) |
| **Gov. Forms (VRDU Reg.)** | VRDU Registration | 50 | 6 | 2 | +0.007 | +0.005 | Near-zero misbinding (well-separated fields) |
| **Gov. Forms (VRDU Ad-buy)** | VRDU Ad-buy | 50 | 9 | 1 | 0.000 | +0.003 | Near-zero misbinding; Haiku perfect binding |

---
---

# Domain I: Financial Documents

> Paystubs and tax forms — structured documents with numeric fields and fixed or semi-fixed layouts.

---

## Experiment 1: Paystubs (Structured Documents)

**Notebook**: `notebooks/hindsight_paystub_cba_v2.ipynb`

### Setup
- **Dataset**: 200 synthetic paystubs with ground-truth annotations (50 sampled for experiment)
- **Method**: Two-stage pipeline — Stage 1 (Tesseract OCR) → Stage 2 (LLM concept mapping)
- **Output format**: Array of `{label, value, concept}` objects
- **Scoring**: Value-first matching (searches for GT values across ALL prediction slots)
- **Eval concepts**: 14 fields across 5 families (compensation, temporal, identity, deductions, tax)

### Results

| Metric | Haiku | GPT-4o-mini |
|--------|-------|-------------|
| Total misbindings | ~1-5 | ~1-5 |
| Delta | ~0 | ~0 |

### Key Finding
Paystubs have **low concept ambiguity** — fields like `gross_pay_current`, `net_pay_current`, `pay_date` are semantically distinct. Only 1-5 misbindings out of ~700 field evaluations. This confirms that **F1 is sufficient for unambiguous domains** and motivated the pivot to more challenging datasets.

---

## Experiment 6: W-2 Tax Forms (Fixed-Layout Documents)

**Notebook**: `notebooks/hindsight_w2_cba.ipynb`

### Setup
- **Dataset**: Synthetic W-2 forms from HuggingFace (`singhsays/fake-w2-us-tax-form-dataset`) — 100+ forms with full ground truth
- **Method**: Vision extraction — LLM reads W-2 images, maps to 44 canonical concepts
- **Eval concepts**: 44 concepts across 8 families (identity, address, compensation, withholding, benefits, box12, box13, state_local)
- **Sample**: 50 W-2 forms
- **Hypothesis**: Fixed IRS layout with labeled boxes should serve as control condition with low misbinding

### Confusable Concept Families

| Family | Concepts | Why Confusable |
|--------|----------|----------------|
| **Compensation** (5) | wages_tips_compensation, social_security_wages, medicare_wages, social_security_tips, allocated_tips | Five dollar-amount wage fields in adjacent boxes |
| **Withholding** (3) | federal_tax_withheld, social_security_tax, medicare_tax | Three tax amounts in the same column |
| **Identity** (5) | employee_ssn, employer_ein, employer_name, employee_name, control_number | SSN (XXX-XX-XXXX) vs EIN (XX-XXXXXXX) similar formats |
| **State/Local** (14) | state_1/2, state_1/2_wages, state_1/2_income_tax, local_1/2_wages, local_1/2_income_tax, local_1/2_name | 14 fields crammed into bottom section, identical structure for line 1 vs line 2 |
| **Box 12** (8) | box_12a-d_code, box_12a-d_value | Four identical code+amount pairs stacked vertically |
| **Benefits** (2) | dependent_care_benefits, nonqualified_plans | Adjacent boxes with similar amounts |

### Results

| Model | Field F1 | CBA-strict | CBA-soft | Delta | Misbindings | Errors |
|-------|----------|-----------|----------|-------|-------------|--------|
| Haiku | 0.420 | 0.292 | 0.348 | **+0.128** | 243 | 0 |
| GPT-4o-mini | 0.587 | 0.493 | 0.530 | **+0.094** | 178 | 0 |
| **Combined** | | | | | **421** | |

### Top Confusion Pairs

| Ground Truth | Predicted As | Count | Family |
|-------------|-------------|:-----:|--------|
| social_security_tips | social_security_wages | 69 | compensation → compensation (SAME) |
| allocated_tips | medicare_wages | 24 | compensation → compensation (SAME) |
| employee_ssn | employer_ein | 21 | identity → identity (SAME) |
| state_2 | state_1 | 20 | state_local → state_local (SAME) |
| state_2_wages | state_1_wages | 17 | state_local → state_local (SAME) |
| local_1_income_tax | state_1_income_tax | 13 | state_local → state_local (SAME) |
| state_2_income_tax | state_1_income_tax | 11 | state_local → state_local (SAME) |
| employer_ein | employee_ssn | 10 | identity → identity (SAME) |
| medicare_tax | social_security_tax | 9 | withholding → withholding (SAME) |

### Family-Level Confusion

| GT Family | Pred Family | Count | Type |
|-----------|-------------|:-----:|------|
| state_local | state_local | 133 | within-family |
| compensation | compensation | 130 | within-family |
| identity | identity | 39 | within-family |
| box12 | box12 | 26 | within-family |
| withholding | withholding | 16 | within-family |
| state_local | withholding | 18 | cross-family |
| compensation | withholding | 14 | cross-family |

**93% of all misbindings are within-family** — models confuse concepts that share the same data type and family.

### Key Finding
**The control condition hypothesis was wrong in a revealing way.** W-2's fixed IRS layout did NOT prevent misbinding — 421 total misbindings with delta up to +0.128. The driver is **concept density**: 44 concepts with many same-type fields (5 wage amounts, 3 tax amounts, 14 state/local fields) create genuine binding ambiguity even when layout is fixed. The single biggest confusion — social_security_tips → social_security_wages (69 occurrences) — shows that adjacent boxes with similar labels and dollar amounts are routinely swapped.

This strengthens the paper's thesis: **misbinding is driven by concept ambiguity (many same-type fields), not layout complexity.** A rigid form with labeled boxes still produces hundreds of misbindings when the concept space is dense.

---
---

# Domain II: Receipts

> Semi-structured commercial documents — variable layouts, multiple numeric fields, OCR challenges.

---

## Experiment 2: CORD Receipts (Indonesian Receipts)

**Notebook**: `notebooks/hindsight_cord_cba.ipynb`

### Setup
- **Dataset**: CORD v2 — 1,000 Indonesian receipts, 30 semantic classes (CC-BY-4.0)
- **Method**: Dual-track design
  - Track A: Tesseract OCR (`lang=ind`) → LLM concept mapping
  - Track B: LLM Vision (direct image → concept mapping)
- **Eval concepts**: 13 fields across 3 families (amounts, payment, counts)
- **Sample**: 100 receipts (filtered for >= 3 eval concepts present)

### Confusable Concept Families

| Family | Concepts | Why Confusable |
|--------|----------|----------------|
| **Amounts** | `subtotal_price`, `tax_price`, `total_price`, `total_etc`, `subtotal_etc`, `tax_etc` | Multiple numeric amounts per receipt, Indonesian labels (Sub Total, Total, Pajak/PPN) |
| **Payment** | `cashprice`, `changeprice`, `creditcardprice`, `emoneyprice`, `otherprice` | Payment methods with similar numeric format (Tunai, Kembali, Bayar) |
| **Counts** | `menuqty_cnt`, `itemsubtotal_cnt` | Both are integer counts (Qty, Jumlah) |

### Results

| Model | Track | Field F1 | CBA-strict | CBA-soft | Delta | Misbindings |
|-------|-------|----------|-----------|----------|-------|-------------|
| Haiku | Tesseract+LLM | 0.4822 | 0.5812 | 0.5946 | -0.0989 | 16 |
| Haiku | LLM Vision | 0.8832 | 0.9022 | 0.9171 | -0.0190 | 15 |
| GPT-4o-mini | Tesseract+LLM | 0.3853 | 0.5455 | 0.5553 | -0.1602 | 15 |
| GPT-4o-mini | LLM Vision | 0.8077 | 0.8111 | 0.8218 | -0.0034 | 7 |
| **Total** | | | | | | **53** |

Sample: 93 of 100 documents met the ≥3-concept threshold.

### Key Finding
**All four configurations show negative delta**: extraction accuracy is the bottleneck, not concept disambiguation. Vision track (Track B) shows that with good extraction (F1 = 80–88%), concept binding is also reliable (CBA = 81–90%). OCR track (Track A) has far larger negative delta (−10% to −16%) driven by Tesseract failures on Indonesian text. Total 53 misbindings across all configurations; 22 on vision track alone. Top confusions: subtotal_price ↔ total_price (amounts family), emoneyprice → otherprice (payment family).

**Root causes of extraction failure (Track A)**: Indonesian number formats (`45.000` = 45,000), currency prefixes (`Rp.`), and OCR quality on receipt images.

---

## Experiment 5: SROIE Receipts (English Receipts)

**Notebook**: `notebooks/hindsight_sroie_cba.ipynb`

### Setup
- **Dataset**: SROIE (ICDAR 2019) via HuggingFace (`sizhkhy/SROIE`) — receipt images with 4 labeled fields
- **Method**: Vision extraction — LLM reads receipt images, maps to 14 canonical concepts
- **Eval concepts**: 14 concepts across 3 families, but only **4 GT-backed** (store_name, store_address, transaction_date, total)
- **Sample**: 50 receipts

### Confusable Concept Families

| Family | Concepts | Why Confusable |
|--------|----------|----------------|
| **Merchant** | store_name, store_address, store_city_state_zip, store_phone | Address vs name on receipt header |
| **Transaction** | transaction_date, transaction_time, receipt_number, cashier, payment_method | Multiple metadata fields |
| **Amounts** | subtotal, tax_amount, total, amount_tendered, change_due | Multiple dollar amounts per receipt |

### Results

| Model | Field F1 | CBA-strict | Delta | Misbindings | Errors |
|-------|----------|-----------|-------|-------------|--------|
| Haiku | 0.705 | 0.705 | **0.000** | 0 | 0 |
| GPT-4o-mini | 0.720 | 0.705 | **+0.015** | 3 | 0 |
| **Combined** | | | | **3** | |

### Top Confusion Pairs

| Ground Truth | Predicted As | Count | Family |
|-------------|-------------|:-----:|--------|
| total | subtotal | 2 | amounts → amounts (SAME) |
| total | amount_tendered | 1 | amounts → amounts (SAME) |

### Key Finding
**Near-zero delta** with only **3 misbindings** (all from GPT-4o-mini). Haiku achieves perfect concept binding (0 misbindings). The low F1 (~0.70) is driven by extraction failures (address matching, receipt OCR quality), not misbinding. SROIE confirms that **receipts with few GT concepts are a low-ambiguity domain** where F1 ≈ CBA. The 3 misbindings that do occur are semantically meaningful: total → subtotal and total → amount_tendered (all within the amounts family).

---
---

# Domain III: Legal Contracts

> Free-form legal prose — semantically overlapping categories, multi-label clauses, highest misbinding rates.

---

## Experiment 3: LEDGAR Employment Law (SEC Contract Provisions)

**Notebook**: `notebooks/hindsight_ledgar_cba.ipynb`

### Setup
- **Dataset**: LEDGAR via LexGLUE (`coastalcph/lex_glue`) — 80,000+ SEC-filed contract provisions, expert-labeled
- **Method**: Bundle approach — 50 bundles of 15 provisions each (one per concept), shuffled. LLM classifies each provision.
- **Eval concepts**: 15 categories across 6 families
- **Scale**: 50 bundles × 15 provisions = 750 classifications per model

### Confusable Concept Families

| Family | Concepts | Why Confusable |
|--------|----------|----------------|
| **Termination Events** | Death, Disability, Terminations | Clauses often mention multiple events ("death, disability, or termination") |
| **Compensation** | Base Salary, Benefits | Compensation clauses reference both salary and benefits |
| **Tax & Withholding** | Tax Withholdings, Taxes, Withholdings | 94% of "Withholdings" mention "tax"; near-synonym labels |
| **Indemnity** | Indemnifications, Indemnity | Essentially the same concept, different grammatical form |
| **Equity** | Vesting, Forfeitures | Equity provisions discuss both vesting schedules and forfeiture conditions |
| **Role** | Employment, Duties, Positions | 66% of "Positions" mention "employment"; overlapping scope |

### Results

| Model | Field F1 | CBA-strict | CBA-soft | Delta | Misbindings | Errors |
|-------|----------|-----------|----------|-------|-------------|--------|
| Haiku | 1.000 | 0.793 | 0.877 | **+0.207** | 155 | 0 |
| GPT-4o-mini | 0.999 | 0.739 | 0.845 | **+0.260** | 195 | 0 |
| **Combined** | | | | | **350** | |

### Per-Concept Accuracy (Lowest CBA-strict)

The most confused concepts:
- **Withholdings** → frequently misclassified as Tax Withholdings (near-synonym)
- **Indemnity** ↔ **Indemnifications** (bidirectional confusion)
- **Death** ↔ **Disability** (co-occurrence in termination clauses)
- **Positions** → confused with Employment, Duties (overlapping HR language)

### Key Finding
**Strong positive delta** (+0.207 to +0.260) with **350 total misbindings**. Both models achieve near-perfect extraction (F1 ≈ 1.0) but systematically confuse semantically overlapping legal concepts. GPT-4o-mini shows weaker concept binding than Haiku (CBA 0.739 vs 0.793).

---

## Experiment 4: CUAD Commercial Law (M&A Due Diligence)

**Notebook**: `notebooks/hindsight_cuad_cba.ipynb`

### Setup
- **Dataset**: CUAD v1 — 510 M&A due diligence contracts, 41 categories, 13,823 expert annotations (SQuAD format)
- **Method**: Bundle approach — 50 bundles of 15 clause spans each. Short spans (< 100 chars, mostly dates) enriched with `>>>context markers<<<` from surrounding text.
- **Eval concepts**: 15 categories across 5 families
- **Scale**: 50 bundles × 15 clauses = 750 classifications per model

### Confusable Concept Families

| Family | Concepts | Why Confusable |
|--------|----------|----------------|
| **Dates** | Agreement Date, Effective Date, Expiration Date | Short date strings, often same date serves dual purposes |
| **License** | License Grant, Non-Transferable License, Irrevocable Or Perpetual License, Exclusivity | Single clause may simultaneously be exclusive, non-transferable, and irrevocable |
| **Liability** | Cap On Liability, Uncapped Liability, Liquidated Damages | Similar boilerplate language, distinction is presence/absence of cap |
| **Term** | Termination For Convenience, Renewal Term, Notice Period To Terminate Renewal | Renewal clauses typically contain notice period requirements |
| **Control** | Change Of Control, Anti-Assignment | Both restrict ownership/control changes |

### Results

| Model | Field F1 | CBA-strict | CBA-soft | Delta | Misbindings | Errors |
|-------|----------|-----------|----------|-------|-------------|--------|
| Haiku | 1.000 | 0.756 | 0.858 | **+0.244** | 183 | 0 |
| GPT-4o-mini | 1.000 | 0.665 | 0.795 | **+0.335** | 251 | 0 |
| **Combined** | | | | | **434** | |

### Top Confusion Pairs

| Ground Truth | Predicted As | Haiku Count | GPT-4o-mini Count |
|-------------|-------------|-------------|-------------------|
| Uncapped Liability | Cap On Liability | 15 | 27 |
| Renewal Term | Notice Period To Terminate Renewal | high | high |
| Notice Period To Terminate Renewal | Renewal Term | bidirectional | bidirectional |
| Effective Date | Agreement Date | significant | significant |
| Exclusivity | License Grant | moderate | moderate |
| Change Of Control | Anti-Assignment | moderate | moderate |

### Key Finding
**Strongest misbinding signal of all domains** — 434 total misbindings with delta up to +0.335. Both models achieve perfect extraction (F1 = 1.0) but CUAD's overlapping legal categories (especially liability and license families) create genuine concept binding challenges. The multi-label nature of legal clauses (same text annotated with multiple categories in CUAD) is a fundamental source of ambiguity.

---
---

# Cross-Domain Analysis

---

## Insights

1. **CBA reveals a hidden failure mode**: Across legal and tax form domains, F1 overstates performance by 9-34%. Traditional metrics completely miss hundreds of concept binding errors.

2. **Two drivers of misbinding**: (a) **Semantic overlap** — legal domains where concepts share overlapping language (LEDGAR, CUAD). (b) **Concept density** — forms with many same-type fields (W-2 with 44 concepts). Both produce significant misbinding regardless of layout structure.

3. **Fixed layout does NOT prevent misbinding**: The W-2 experiment disproves the hypothesis that structured forms with labeled boxes are immune to misbinding. With 44 concepts and many same-type fields, W-2 produces 421 misbindings — more than LEDGAR (350).

4. **93% within-family confusion on W-2**: Almost all W-2 misbindings occur between concepts of the same family (wage₁ ↔ wage₂, tax₁ ↔ tax₂), showing that concept density within a family is the key driver.

5. **Model comparison**: GPT-4o-mini consistently shows weaker concept binding than Haiku on legal tasks (CBA 0.665 vs 0.756 on CUAD), but stronger on W-2 (CBA 0.493 vs 0.292), suggesting different strengths in text vs vision binding.

6. **Negative delta is informative too**: CORD's negative delta correctly identifies a domain where extraction (not binding) is the bottleneck — CBA doesn't artificially inflate problems.

7. **Low-ambiguity baselines validate the metric**: Paystubs (~0 delta) and SROIE (3 misbindings) confirm CBA doesn't falsely flag misbinding in unambiguous domains.

---

## Reproducibility

All experiments use:
- **Scoring**: Value-first matching (searches for GT values across all prediction slots before checking concept keys)
- **Output format**: JSON with canonical concept keys
- **Random seed**: 42 (for sampling and bundle construction)
- **API keys**: Via environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
- **Rate limiting**: 5s sleep between OpenAI vision requests, 429-aware exponential backoff (10s → 20s → 40s → 80s)
- **Public datasets**: CORD v2, LEDGAR (via LexGLUE), CUAD v1, SROIE (ICDAR 2019), singhsays/fake-w2-us-tax-form-dataset — all freely available

To reproduce, set API keys and run the notebooks in order. Each notebook is self-contained with its own data loading, extraction, scoring, and visualization.
