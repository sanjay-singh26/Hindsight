# Hindsight: Experiment Phase Documentation

## Overview

Hindsight is a research project investigating **Concept-Binding Accuracy (CBA)** as an evaluation metric for document intelligence systems. The core hypothesis is that existing field-level accuracy metrics (e.g., field F1) overstate model performance by failing to detect *misbindings* --- cases where a model extracts the correct value but assigns it to the wrong semantic concept.

The research spans four document domains (paystubs, invoices/receipts, contracts, and W-2 tax forms) and is organized into six sequential phases over a 12-week timeline. Each phase has explicit deliverables, success criteria, and go/no-go decision points.

**Key Metric Definitions:**

| Metric | Definition |
|--------|------------|
| **Field F1** | Standard precision/recall over extracted field values, ignoring concept identity |
| **CBA-strict** | Fraction of extractions where both the value is correct AND it is bound to the correct canonical concept |
| **CBA-soft** | Relaxed CBA that awards partial credit for values bound to a concept within the same semantic family |
| **CBA-weighted** | CBA variant that weights misbindings by their downstream severity (domain-specific) |
| **Delta** | `Field F1 - CBA-strict`; the misbinding gap. Higher delta = more concept confusion |

---

## Phase 0 --- Pilot Validation

| Attribute | Detail |
|-----------|--------|
| **Timeline** | Week 1 |
| **Status** | COMPLETED |

### Objective

Confirm the core hypothesis --- that a measurable gap exists between field-level accuracy and concept-binding accuracy --- before committing to the full 12-week research program. This phase serves as a rapid feasibility check.

### Inputs

- 10 diverse synthetic paystub documents (varying layouts, employers, pay frequencies)
- Two frontier models: Claude Sonnet 3.5 and GPT-5
- A simple extraction prompt requesting approximately 14 canonical concept fields (employee_name, employer_name, pay_date, pay_period_start, pay_period_end, pay_frequency, gross_pay_current, net_pay_current, gross_pay_ytd, net_pay_ytd, federal_tax_withheld, state_tax_withheld, social_security_tax, medicare_tax)

### Method

1. Generate 10 synthetic paystubs using the template engine with diverse layouts
2. Define ground-truth annotations for all 14 concepts per document
3. Send each document to Claude and GPT with a structured extraction prompt using canonical concept names as JSON keys
4. Score responses using field F1 and CBA-strict
5. Compute delta (field F1 - CBA-strict) per document and overall

### Outputs / Deliverables

- `results/hindsight_phase0_results.json` --- per-document and aggregate scores, raw predictions, and misbinding log
- Go/no-go decision based on observed delta

### Success Criteria

| Criterion | Threshold | Observed |
|-----------|-----------|----------|
| Kill signal | Delta < 5% across all experiments would trigger project reconsideration | **Delta = 0.0%** (Sonnet), **0--2.2%** across all runs |
| Target range | Delta between 20--40% would strongly validate the hypothesis | **Not reached** |

### Results and Honest Assessment

Phase 0 produced uniformly low deltas:

- **Claude Sonnet on 10 paystubs**: Field F1 = 1.0, CBA-strict = 1.0, Delta = 0.0 (zero misbindings across all 10 documents)
- **Subsequent adversarial run (Phase 1)**: Field F1 = 1.0, CBA-strict = 0.9929, Delta = 0.0071 (one misbinding in adv_09: pay_date confused with pay_period_end)

The observed delta of 0--2.2% is **below the kill signal of 5%**. However, this does not necessarily invalidate the hypothesis. Several methodological factors likely suppressed delta:

1. **Synthetic documents with clean rendering**: Generated paystubs had unambiguous labels and no OCR noise
2. **Simple prompt with canonical keys**: Using exact concept names as JSON keys essentially "tells" the model the answer schema
3. **Single domain (paystubs)**: Paystubs have relatively structured layouts; contracts and receipts may exhibit much higher ambiguity
4. **Small sample size**: 10 documents is insufficient to surface rare misbindings

**Decision**: Proceed to Phase 1 with methodology adjustments --- introduce adversarial layouts, noisy scans, and additional domains to stress-test the metric.

---

## Phase 1 --- Concept Ontology Design

| Attribute | Detail |
|-----------|--------|
| **Timeline** | Weeks 1--2 |
| **Status** | IN PROGRESS --- YAML ontologies created, awaiting expert validation |

### Objective

Finalize the canonical concept identifiers, semantic family structures, confusability maps, and annotation guidelines for all four benchmark domains. The ontology defines what "correct concept binding" means and is the foundation for all subsequent scoring.

### Inputs

- Domain expertise (payroll, accounting, legal, tax)
- Existing field schemas from source datasets (SROIE 4-field schema, CUAD 41-category schema, IRS W-2 box schema)
- Pilot results from Phase 0 identifying initial confusion pairs

### Deliverables Per Domain

For each of the four domains, produce:

1. **Concept Catalog** --- YAML file listing every canonical concept with unique `id`, human-readable `name`, precise `definition`, and `confusable_with` list
2. **Semantic Family Structure** --- Grouping of concepts into families (e.g., compensation, temporal, identity, deductions, address) for CBA-soft scoring
3. **Expected Confusion Matrix** --- Prior hypotheses about which concept pairs models will confuse, derived from structural and linguistic similarity
4. **Annotation Guidelines** --- Per-domain rules for assigning concept labels to extracted values (see `docs/annotation_guidelines.md`)

### Domain Details

#### Paystubs (27 concepts) --- PRIMARY DOMAIN

**Status**: Ontology complete (`data/paystubs/ontology.yaml`)

Five semantic families with 27 canonical concepts:

| Family | Concepts | Count |
|--------|----------|-------|
| Compensation | gross_pay_current, net_pay_current, regular_pay_current, overtime_pay_current, bonus_current, gross_pay_ytd, net_pay_ytd, total_deductions_current, total_deductions_ytd | 9 |
| Temporal | pay_date, pay_period_start, pay_period_end, pay_frequency | 4 |
| Identity | employee_name, employer_name, employee_id, ssn_last4 | 4 |
| Deductions | federal_tax_withheld, state_tax_withheld, social_security_tax, medicare_tax, retirement_401k, health_insurance | 6 |
| Address | employee_address, employee_city_state_zip, employer_address, employer_city_state_zip | 4 |

Key predicted confusion pairs: gross_pay_current/net_pay_current, pay_date/pay_period_end, employer_address/employee_address, social_security_tax/medicare_tax, health_insurance/retirement_401k.

#### Invoices / Receipts (~20 concepts)

**Status**: Initial analysis from SROIE; ontology draft needed

SROIE provides only 4 fields (company, date, address, total), each of which is itself ambiguous at the concept level:

| SROIE Field | Possible Canonical Concepts |
|-------------|----------------------------|
| company | store_name, parent_company, franchise_name, brand_name |
| date | transaction_date, print_date, receipt_date |
| address | store_address, headquarters_address |
| total | subtotal_pretax, total_with_tax, amount_due, amount_tendered, grand_total |

The ontology must expand SROIE's 4 fields to approximately 20 canonical concepts covering line items, tax breakdowns, payment methods, and merchant identifiers.

#### Contracts (~18 concepts) --- LEGAL EXPERT LEADS

**Status**: Expert guide delivered (`CUAD_CBA_Guide_For_Legal_Expert.md`); awaiting expert response

From CUAD's 41 categories, approximately 18 canonical concepts will be selected, organized into families:

| Family | Proposed Concepts |
|--------|-------------------|
| Parties & Identification | primary_party, counterparty |
| Dates & Duration | agreement_date, effective_date, expiration_date, renewal_term, notice_period_to_terminate |
| Restrictive Covenants | non_compete, exclusivity, no_solicit_customers, no_solicit_employees, non_disparagement |
| IP & Licensing | ip_ownership_assignment, license_grant, license_transferability |
| Liability & Remedies | liability_cap, liquidated_damages, insurance_requirement |
| Termination & Control | termination_for_convenience, change_of_control_trigger |

High-confusion pairs identified for legal validation: agreement_date / effective_date, non_compete / exclusivity, no_solicit_customers / no_solicit_employees, affiliate_license_licensor / affiliate_license_licensee, uncapped_liability / cap_on_liability.

#### W-2 Tax Forms (~15 concepts) --- CONTROL DOMAIN

**Status**: Dataset preparation started (`hindsight_w2_dataset_prep.ipynb`); ontology draft needed

W-2 forms have a fixed grid layout with numbered boxes, making them the lowest-ambiguity domain and therefore the ideal control:

| Family | Concepts |
|--------|----------|
| Wages | wages_tips_compensation, social_security_wages, medicare_wages |
| Taxes | federal_tax_withheld, social_security_tax, medicare_tax |
| Identity | employee_ssn, employer_ein, employee_name, employer_name |
| Address | employee_address, employer_address |
| Supplemental | box_12a_value, box_12b_value, state_1_wages, state_1_income_tax |

Predicted delta for W-2: low (< 3%) due to rigid layout and box-number cues.

### Success Criteria

| Criterion | Description |
|-----------|-------------|
| Completeness | All four domain ontologies defined with >= 15 concepts each |
| Expert validation | Contract ontology reviewed by legal domain expert |
| Confusability coverage | Each ontology has >= 5 predicted confusion pairs documented |
| Inter-annotator test | Two annotators achieve >= 90% agreement on concept assignment for 10 sample documents |

### Estimated Cost

$0 (labor only; no API costs)

---

## Phase 2 --- Dataset Construction

| Attribute | Detail |
|-----------|--------|
| **Timeline** | Weeks 2--4 |
| **Status** | PARTIALLY DONE --- 200 paystubs generated, SROIE analysis started, W-2 prep started |

### Objective

Build a multi-domain document corpus of 300--530 documents, each annotated at the concept level with canonical concept IDs, ground-truth values, and ambiguity ratings. The dataset must include realistic visual noise and layout variation to stress-test vision-language models.

### Inputs

- Finalized ontologies from Phase 1
- Source datasets: SROIE (ICDAR 2019), CUAD (Atticus Project), singhsays/fake-w2-us-tax-form-dataset (HuggingFace)
- Paystub template engine (`paystub_generator/`)

### Per-Domain Construction

#### Paystubs: 100--200 synthetic documents

**Method**: Template engine generates paystub images from parameterized templates.

| Parameter | Variation |
|-----------|-----------|
| Layouts | 15--20 distinct templates (left_header, wide_format, compact, retail, check_style, detailed_stub, etc.) |
| Surface labels | Varied per template (e.g., "Gross Pay" vs "Total Earnings" vs "Gross Compensation") |
| Noise types | clean, light_scan, heavy_scan, phone_photo |
| Pay frequencies | weekly, biweekly, semi-monthly, monthly |
| Edge cases | state_tax = $0.00, pay_date = pay_period_end (same value), very similar gross/net amounts |

**Status**: 200 paystubs generated with ground-truth JSON annotations per document.

#### Invoices / Receipts: 100--150 documents

**Source**: SROIE dataset (ICDAR 2019) and FATURA dataset

**Method**: Re-annotate existing OCR-extracted documents with concept-level labels. SROIE provides 4 coarse fields; the task is to map each value to the appropriate canonical concept from the expanded ontology.

**Status**: SROIE analysis notebook created (`hindsight_sroie_analysis.ipynb`); concept-level re-annotation in progress.

#### Contracts: 50--80 documents

**Source**: CUAD dataset (510 contracts, 41 clause categories)

**Method**: Select 50--80 contracts in consultation with legal expert. Re-map CUAD's 41-category annotations to the reduced 18-concept ontology. Legal expert validates mappings and flags disagreements (which become ambiguity annotations).

**Status**: Expert guide delivered; awaiting contract selection and validation.

#### W-2 Tax Forms: 50--100 documents

**Source**: singhsays/fake-w2-us-tax-form-dataset (HuggingFace)

**Method**: Download synthetic W-2 images, extract ground-truth from metadata, and annotate with canonical concept IDs. Add noise variants (scan artifacts, rotation) to a subset.

**Status**: Dataset preparation notebook created (`hindsight_w2_dataset_prep.ipynb`); download and annotation pipeline in progress.

### Outputs / Deliverables

| Deliverable | Format | Description |
|-------------|--------|-------------|
| Document images | PNG/JPEG | One image per document, organized by domain |
| Ground-truth annotations | JSON per document | `{concept_id: value}` mapping |
| Ambiguity annotations | JSON per document | Per-field ambiguity level (0--3) and alternative concept list |
| Manifest | `manifest.json` | Index of all documents with metadata (domain, layout, noise, split) |
| Datasheet | `docs/datasheet.md` | Gebru et al. datasheet for the combined dataset |

### Total Corpus Summary

| Domain | Target Count | Status | Source |
|--------|-------------|--------|--------|
| Paystubs | 100--200 | 200 generated | Synthetic (template engine) |
| Invoices/Receipts | 100--150 | In progress | SROIE + FATURA |
| Contracts | 50--80 | Awaiting expert | CUAD |
| W-2 Forms | 50--100 | In progress | HuggingFace synthetic |
| **Total** | **300--530** | **~200 complete** | |

### Success Criteria

| Criterion | Description |
|-----------|-------------|
| Volume | >= 300 documents across all four domains |
| Diversity | >= 10 distinct layouts per domain (where applicable) |
| Annotation quality | Double-annotation on 10% subset with >= 90% inter-annotator agreement |
| Noise coverage | >= 30% of documents include visual noise (scan artifacts, phone photos) |

### Estimated Cost

$0 (public datasets and synthetic generation; no licensing fees)

---

## Phase 3 --- Evaluation Library

| Attribute | Detail |
|-----------|--------|
| **Timeline** | Weeks 3--5 |
| **Status** | IN PROGRESS --- package skeleton created (`hindsight_eval/`) |

### Objective

Build `hindsight-eval` as a pip-installable Python package that computes all CBA metrics from standardized input/output formats. The library must be domain-agnostic (works with any ontology YAML) and produce both scalar scores and diagnostic artifacts (confusion matrices, misbinding logs).

### Inputs

- Ontology YAML files (from Phase 1)
- Ground-truth annotations (from Phase 2)
- Model prediction JSON files

### Package Design

```
hindsight-eval/
  hindsight_eval/
    __init__.py
    metrics.py       # CBA-strict, CBA-soft, CBA-weighted, field F1, delta
    ontology.py      # Load and validate ontology YAML
    scorer.py        # End-to-end scoring pipeline
    confusion.py     # Confusion matrix generation
    cli.py           # Command-line interface
  pyproject.toml
```

**Installation**: `pip install -e .` (or `pip install hindsight-eval` after PyPI publication)

**CLI usage**: `hindsight score --ontology data/paystubs/ontology.yaml --predictions results/run.json --ground-truth data/paystubs/gt/`

### Metric Specifications

#### CBA-strict

For each document, compute the fraction of extracted fields where:
1. The predicted value matches the ground-truth value (exact or fuzzy match), AND
2. The predicted concept key matches the ground-truth concept key

```
CBA-strict = |{f : value_match(f) AND concept_match(f)}| / |F|
```

#### CBA-soft

Same as CBA-strict but awards partial credit when the predicted concept is in the same semantic family as the ground-truth concept:

```
CBA-soft(f) = 1.0    if exact concept match
            = alpha   if same-family concept (default alpha = 0.5)
            = 0.0    if different family
```

#### CBA-weighted

Weights misbindings by domain-specific severity. For paystubs, confusing gross_pay with net_pay is more consequential than confusing employee_address with employer_address:

```
CBA-weighted = sum(w_f * match(f)) / sum(w_f)
```

Weights are defined per-domain in the ontology YAML.

#### Field F1

Standard precision/recall over extracted values, ignoring concept identity. Used as the baseline for delta computation.

#### Delta

```
delta = field_f1 - cba_strict
```

A positive delta indicates that field-level evaluation overstates model accuracy by failing to penalize misbindings.

### Input / Output Formats

**Input (predictions)**:
```json
{
  "document_id": "ps_0061",
  "predictions": {
    "concept_id": "extracted_value",
    ...
  }
}
```

**Input (ground truth)**:
```json
{
  "document_id": "ps_0061",
  "ground_truth": {
    "concept_id": "correct_value",
    ...
  },
  "metadata": {
    "domain": "paystub",
    "layout": "left_header",
    "noise": "heavy_scan"
  }
}
```

**Output (scores)**:
```json
{
  "field_f1": 0.8859,
  "cba_strict": 0.8807,
  "cba_soft": 0.9134,
  "cba_weighted": 0.8952,
  "delta": 0.0052,
  "num_misbindings": 7,
  "misbindings": [
    {
      "doc_id": "ps_0061",
      "concept": "employer_address",
      "gt_value": "147 birch court",
      "pred_value": "347 main circuit",
      "bound_to": "employee_address"
    }
  ]
}
```

### Success Criteria

| Criterion | Description |
|-----------|-------------|
| Installability | `pip install -e .` succeeds on Python >= 3.9 |
| Correctness | Unit tests cover all metric computations; hand-verified on Phase 0/1 results |
| Domain-agnostic | Accepts any ontology YAML; tested on paystub and W-2 ontologies |
| CLI functional | `hindsight score` command produces JSON output from CLI |
| Reproducibility | Scoring the Phase 0 results file produces identical numbers to notebook computations |

### Estimated Cost

$0 (development effort only)

---

## Phase 4 --- Model Evaluation

| Attribute | Detail |
|-----------|--------|
| **Timeline** | Weeks 5--8 |
| **Status** | PARTIALLY DONE --- ran Sonnet/Haiku/GPT-4o/GPT-4o-mini on paystubs and SROIE |

### Objective

Evaluate a broad set of document intelligence models across all four domains using the standardized extraction prompt and CBA scoring pipeline. The goal is to produce the main results table for the paper: field F1 vs. CBA-strict vs. delta, broken down by model, domain, and concept family.

### Model Categories

#### Frontier Vision-Language Models (Estimated API cost: $50--80)

| Model | Provider | Estimated Cost |
|-------|----------|---------------|
| Claude Opus | Anthropic | $15--25 |
| Claude Sonnet | Anthropic | $10--15 |
| Claude Haiku | Anthropic | $3--5 |
| GPT-4o | OpenAI | $10--15 |
| GPT-4o-mini | OpenAI | $3--5 |
| Gemini 1.5 Pro | Google | $5--10 |
| Gemini 1.5 Flash | Google | $2--3 |

#### Open-Source Vision-Language Models (Free --- local inference)

| Model | Parameters | Notes |
|-------|-----------|-------|
| LLaVA-1.6 | 34B | Strong document understanding |
| Qwen-VL-Plus | 72B | Competitive on DocVQA |
| InternVL2 | 26B | SOTA on several benchmarks |

#### Commercial Document AI (Estimated API cost: $30--50)

| Service | Provider |
|---------|----------|
| Amazon Textract | AWS |
| Azure Document Intelligence | Microsoft |
| Google Document AI | Google Cloud |

#### Pipeline / OCR Tools (Free)

| Tool | Notes |
|------|-------|
| Tesseract + heuristics | Baseline OCR pipeline |
| PaddleOCR + extraction | Open-source alternative |
| DocTR | Hugging Face document toolkit |

### Experimental Protocol

For each model-domain pair:

1. **Prompt construction**: Use a standardized extraction prompt per domain. The prompt lists all canonical concept IDs from the ontology as the expected JSON keys. No definitions or examples are provided (zero-shot).

2. **Inference**: Send document image + prompt to the model. Collect structured JSON response.

3. **Scoring**: Run `hindsight-eval` to compute field F1, CBA-strict, CBA-soft, CBA-weighted, delta, and per-concept confusion matrix.

4. **Ablations**:
   - **Prompt sensitivity**: Full prompt (with definitions) vs. minimal prompt (keys only) vs. naive prompt (generic field names)
   - **Noise sensitivity**: Clean vs. noisy document variants
   - **Layout sensitivity**: Score stratified by layout type

### Results Obtained So Far

#### Paystub Results (Phase 2--3 experiments)

| Experiment | Model | Field F1 | CBA-strict | Delta | Misbindings |
|------------|-------|----------|------------|-------|-------------|
| Phase 0 (10 clean paystubs) | Sonnet | 1.000 | 1.000 | 0.000 | 0 |
| Phase 1 (10 adversarial) | Sonnet | 1.000 | 0.993 | 0.007 | 1 |
| Phase 2 (10 adversarial, full prompt) | Sonnet | 1.000 | 1.000 | 0.000 | 0 |
| Phase 3 (50 image paystubs) | Sonnet | 0.886 | 0.881 | 0.005 | 7 |
| Phase 3 revised (50 image paystubs) | Sonnet | 0.885 | 0.879 | 0.007 | 9 |

Notable misbinding patterns observed:
- **Address swaps**: employer_address bound to employee_address (heavy_scan layouts)
- **Deduction cascade**: state_tax -> social_security_tax -> medicare_tax -> health_insurance (cascading misbindings within the deductions family)
- **Identity confusion**: ssn_last4 bound to employee_id (check_style layouts)
- **Temporal confusion**: pay_period_start bound to pay_date (left_header layouts)
- **Compensation confusion**: regular_pay_current bound to gross_pay_current (retail layouts)

### Outputs / Deliverables

| Deliverable | Description |
|-------------|-------------|
| Results JSON per model-domain | Aggregate and per-document scores with misbinding logs |
| Main results table | LaTeX table: Model x Domain x {Field F1, CBA-strict, Delta} |
| Per-concept confusion matrices | Heatmap showing which concepts each model confuses |
| Ablation results | Impact of prompt type, noise level, and layout on delta |

### Success Criteria

| Criterion | Description |
|-----------|-------------|
| Coverage | >= 5 frontier models, >= 2 commercial doc AI, >= 2 open-source evaluated |
| All domains | Results for all 4 domains (paystubs, invoices, contracts, W-2) |
| Statistical significance | Confidence intervals on delta via bootstrap resampling |
| Reproducibility | All inference scripts and prompts committed to repository |

### Estimated Cost

| Category | Cost Range |
|----------|-----------|
| Frontier VLM API calls | $50--80 |
| Commercial document AI | $30--50 |
| Open-source models | $0 (local GPU) |
| **Total** | **$80--160** |

---

## Phase 5 --- Analysis and Writing

| Attribute | Detail |
|-----------|--------|
| **Timeline** | Weeks 8--12 |
| **Status** | NOT STARTED |

### Objective

Conduct six key analyses on the evaluation results and produce the final research paper in NeurIPS LaTeX format. The paper should present CBA as a complementary evaluation metric and provide empirical evidence for concept-binding errors across domains and models.

### Six Key Analyses

#### Analysis 1: Delta Table

The headline result. A table showing delta (field F1 - CBA-strict) for every model-domain combination.

| Expected Finding | Delta is small (< 3%) for structured domains (W-2) and larger (5--15%) for semi-structured domains (receipts, contracts) |
|-----------------|------|

#### Analysis 2: Confusion Matrices

Per-model, per-domain confusion matrices showing which canonical concepts are most frequently confused. Rows = ground-truth concepts, columns = predicted concepts. Off-diagonal entries reveal systematic misbinding patterns.

| Expected Finding | Confusion clusters around semantic families (e.g., within temporal concepts, within deduction concepts) |
|-----------------|------|

#### Analysis 3: Cross-Domain Clustering

Compare confusion patterns across domains. Do the same "shapes" of confusion appear in paystubs (gross_pay vs. net_pay) and receipts (subtotal vs. total)?

| Expected Finding | Analogous confusion pairs exist across domains, suggesting a general weakness in concept binding rather than domain-specific artifacts |
|-----------------|------|

#### Analysis 4: Ambiguity Gradient

Correlate the ambiguity annotations (level 0--3) from Phase 2 with observed misbinding rates. Documents with higher annotated ambiguity should produce larger deltas.

| Expected Finding | Strong positive correlation (r > 0.5) between annotated ambiguity and observed delta |
|-----------------|------|

#### Analysis 5: CBA-soft vs. CBA-strict

Compare the two CBA variants to understand whether misbindings occur primarily within or across semantic families.

| Expected Finding | CBA-soft >> CBA-strict would indicate most errors are within-family (e.g., social_security_tax confused with medicare_tax, both in the deductions family) |
|-----------------|------|

#### Analysis 6: CBA-weighted

Apply domain-specific severity weights to assess real-world impact of misbindings.

| Expected Finding | CBA-weighted reveals that some misbindings with high delta have low practical impact (e.g., address swaps) while others with low delta have high impact (e.g., gross/net pay confusion in financial underwriting) |
|-----------------|------|

### Paper Structure

Target venue: NeurIPS 2026 (Datasets and Benchmarks track)

| Section | Content | Est. Pages |
|---------|---------|-----------|
| 1. Introduction | Problem statement, CBA definition, motivation | 1 |
| 2. Related Work | Document AI benchmarks, evaluation metrics, concept grounding | 1 |
| 3. Concept-Binding Accuracy | Formal metric definition (CBA-strict, CBA-soft, CBA-weighted), delta | 1.5 |
| 4. Benchmark Construction | Four domains, ontology design, dataset statistics, annotation process | 1.5 |
| 5. Experiments | Models evaluated, prompt design, experimental protocol | 1 |
| 6. Results | Delta table, confusion matrices, six key analyses | 2 |
| 7. Discussion | Implications for document AI evaluation, limitations, ethical considerations | 0.5 |
| 8. Conclusion | Summary of contributions, future work | 0.5 |
| Appendix | Full ontologies, per-document results, additional confusion matrices, datasheet | 3--5 |
| **Total** | | **9 + appendix** |

### Outputs / Deliverables

| Deliverable | Format |
|-------------|--------|
| Research paper | LaTeX (NeurIPS 2026 template), 9 pages + appendix |
| Figures | Delta bar charts, confusion matrix heatmaps, ambiguity correlation scatter plots |
| Supplementary materials | Full results JSON, ontology YAML files, evaluation scripts |
| Code release | `hindsight-eval` package on GitHub (MIT license) |
| Dataset release | Benchmark documents and annotations (license TBD per source dataset) |

### Success Criteria

| Criterion | Description |
|-----------|-------------|
| Paper quality | Accepted at NeurIPS 2026 Datasets and Benchmarks track (or comparable venue) |
| Reproducibility | All results reproducible from released code and data |
| Novel contribution | CBA metric adopted or cited by at least one subsequent document AI evaluation |

### Estimated Cost

$0 (writing and analysis effort only; all API costs incurred in Phase 4)

---

## Budget Summary

| Phase | API Cost | Other Cost | Total |
|-------|----------|-----------|-------|
| Phase 0 --- Pilot | ~$5 | $0 | ~$5 |
| Phase 1 --- Ontology | $0 | $0 (labor) | $0 |
| Phase 2 --- Dataset | $0 | $0 | $0 |
| Phase 3 --- Eval Library | $0 | $0 | $0 |
| Phase 4 --- Model Evaluation | $80--160 | $0 | $80--160 |
| Phase 5 --- Analysis & Writing | $0 | $0 | $0 |
| **Total** | **$85--165** | **$0** | **$85--165** |

---

## Status Dashboard

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 0 --- Pilot Validation | COMPLETED | 100% |
| Phase 1 --- Concept Ontology Design | IN PROGRESS | ~60% (paystub ontology done, contracts guide delivered, invoices/W-2 drafts needed) |
| Phase 2 --- Dataset Construction | PARTIALLY DONE | ~40% (200 paystubs done, SROIE analysis started, W-2 prep started, contracts awaiting expert) |
| Phase 3 --- Evaluation Library | IN PROGRESS | ~30% (package skeleton and pyproject.toml created) |
| Phase 4 --- Model Evaluation | PARTIALLY DONE | ~25% (Sonnet/Haiku on paystubs complete, other models and domains pending) |
| Phase 5 --- Analysis & Writing | NOT STARTED | 0% |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Delta remains < 5% across all domains | Fatal --- no paper | Expand to more ambiguous domains (contracts, medical records); increase document noise; use naive prompts that remove concept definitions |
| Legal expert does not respond | Blocks contract domain | Proceed with 3 domains; contract ontology can be built from CUAD documentation alone (weaker but viable) |
| Open-source models too slow for full corpus | Reduces model coverage | Subsample to 50 documents per domain for open-source models |
| SROIE re-annotation introduces bias | Weakens invoice results | Double-annotation by independent annotators; measure inter-annotator agreement |
| Synthetic paystubs do not generalize | Questions ecological validity | Include real-world document noise; compare with any available real paystub results (privacy-permitting) |
