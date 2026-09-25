# Hindsight: Guide for Adding a New Domain

## Overview

The Hindsight benchmark is designed to be domain-extensible. Any document type that involves structured or semi-structured information extraction can be evaluated using Concept-Binding Accuracy (CBA), provided the domain has a well-defined concept ontology.

This guide describes the four-step process for adding a new domain to the benchmark.

---

## Prerequisites

Before adding a new domain, ensure you have:

- **Domain expertise** (or access to a domain expert) sufficient to define canonical concepts and resolve annotation ambiguities
- **A corpus of documents** (50--200) in the target domain, with either existing field-level annotations or the resources to create them
- **The `hindsight-eval` package** installed (`pip install -e .` from the repository root)
- **Familiarity with the annotation guidelines** (`docs/annotation_guidelines.md`)

---

## Step 1: Define the Concept Ontology

### 1.1 Create the Ontology YAML

Create a YAML file at `data/<domain_name>/ontology.yaml` following the schema used by existing domains. The file must define:

- **Domain identifier**: A short, lowercase name (e.g., `paystub`, `invoice`, `contract`, `w2`)
- **Semantic families**: Logical groupings of related concepts
- **Concepts**: Each concept requires:
  - `id`: A unique, snake_case identifier (e.g., `gross_pay_current`)
  - `name`: A human-readable name (e.g., "Gross Pay (Current)")
  - `definition`: A precise, unambiguous natural-language definition. This is the authoritative reference for annotation and evaluation. Write it as if explaining to a domain expert who has never seen this specific ontology.
  - `confusable_with`: A list of concept IDs that this concept could plausibly be confused with

### 1.2 Example Structure

```yaml
domain: medical_record
families:
  - name: patient_identity
    concepts:
      - id: patient_name
        name: "Patient Full Name"
        definition: >
          The full legal name of the patient as recorded in the medical
          document. This is the individual receiving care, not the
          referring physician, attending physician, or emergency contact.
        confusable_with: [attending_physician_name, referring_physician_name]

      - id: patient_dob
        name: "Patient Date of Birth"
        definition: >
          The patient's date of birth as recorded in the demographic
          section of the document. This is a fixed biographical date,
          not the date of service, admission date, or discharge date.
        confusable_with: [date_of_service, admission_date]

  - name: encounter
    concepts:
      - id: date_of_service
        name: "Date of Service"
        definition: >
          The calendar date on which the medical service or procedure
          was performed. For multi-day encounters, this is the primary
          service date. Distinct from admission date, discharge date,
          and billing date.
        confusable_with: [admission_date, discharge_date, patient_dob]

      - id: admission_date
        name: "Admission Date"
        definition: >
          The date the patient was formally admitted to a healthcare
          facility for inpatient care. This marks the beginning of the
          inpatient stay and is distinct from the date of service (which
          may refer to a specific procedure within the stay).
        confusable_with: [date_of_service, discharge_date]
```

### 1.3 Design Guidelines

**Target 15--30 concepts per domain.** Fewer than 15 may not generate enough confusion pairs for meaningful CBA analysis. More than 30 creates annotation burden without proportional analytical benefit.

**Define at least 3 semantic families.** CBA-soft scoring relies on family structure to distinguish within-family from cross-family misbindings.

**Populate `confusable_with` thoughtfully.** These predictions are tested empirically. Include pairs where:
- The concepts share similar value formats (e.g., two dollar amounts, two dates)
- The concepts often appear near each other on the document
- The concepts have overlapping or similar surface labels across different document variants
- Domain knowledge is required to distinguish them

**Write definitions that are self-contained.** An annotator should be able to assign a value to the correct concept using only the definition text, without needing to compare against other definitions. Avoid circular references (e.g., "not the same as concept X" is acceptable as a clarifying note, but the positive definition must stand alone).

### 1.4 Validation Checklist

Before proceeding to Step 2, verify:

- [ ] Every concept has a unique `id` (no duplicates within or across families)
- [ ] Every `confusable_with` reference points to a valid concept `id` in the same ontology
- [ ] Definitions are precise enough that two independent annotators would agree on concept assignment >= 90% of the time
- [ ] At least 5 concept pairs are identified as potentially confusable
- [ ] The YAML file parses without errors: `python -c "import yaml; yaml.safe_load(open('data/<domain>/ontology.yaml'))"`

---

## Step 2: Annotate Documents with Concept-Level Labels

### 2.1 Prepare the Document Corpus

Organize documents in `data/<domain_name>/`:

```
data/<domain_name>/
  ontology.yaml          # From Step 1
  images/                # Document images (PNG or JPEG)
    doc_0001.png
    doc_0002.png
    ...
  ground_truth/          # One JSON file per document
    doc_0001.json
    doc_0002.json
    ...
  metadata.json          # Corpus-level metadata
```

### 2.2 Annotation Format

Each ground-truth JSON file follows this schema:

```json
{
  "document_id": "doc_0001",
  "domain": "medical_record",
  "source": "synthetic",
  "layout": "standard_ehr",
  "noise": "clean",
  "annotator_id": "annotator_01",
  "annotation_date": "2025-08-01",
  "annotation_version": 1,
  "ground_truth": {
    "patient_name": "Jane M. Doe",
    "patient_dob": "03/15/1985",
    "date_of_service": "07/22/2025",
    "admission_date": "07/20/2025",
    "attending_physician_name": "Dr. Robert Kim"
  },
  "ambiguity": {
    "date_of_service": {
      "level": 1,
      "alternatives": ["admission_date"],
      "cue": "label-based --- clearly labeled 'Service Date' in header",
      "notes": ""
    }
  }
}
```

### 2.3 Annotation Process

Follow the procedures described in `docs/annotation_guidelines.md`:

1. Assign each value to its best-fitting canonical concept using ontology definitions
2. Record ambiguity annotations for any field at level >= 1
3. Run automated consistency checks (schema validation, completeness)
4. Double-annotate a 10% random subset
5. Resolve disagreements per the disagreement protocol
6. Finalize with annotation metadata

### 2.4 Corpus-Level Metadata

Create a `metadata.json` file summarizing the corpus:

```json
{
  "domain": "medical_record",
  "num_documents": 75,
  "num_concepts": 22,
  "num_families": 4,
  "source_datasets": ["synthetic_ehr_generator"],
  "layout_distribution": {
    "standard_ehr": 40,
    "legacy_paper": 20,
    "fax_scan": 15
  },
  "noise_distribution": {
    "clean": 50,
    "light_scan": 15,
    "heavy_scan": 10
  },
  "annotation_statistics": {
    "double_annotated_pct": 10.0,
    "inter_annotator_kappa": 0.88,
    "mean_ambiguity_level": 0.95,
    "pct_fields_ambiguous_level2_plus": 12.3
  }
}
```

### 2.5 Minimum Requirements

| Requirement | Threshold |
|-------------|-----------|
| Document count | >= 50 |
| Layout variety | >= 3 distinct layouts (if applicable to the domain) |
| Noise variety | >= 2 noise levels (clean + at least one degraded) |
| Double annotation | >= 10% of documents |
| Inter-annotator kappa | >= 0.80 |

---

## Step 3: Run `hindsight-eval` on the New Domain

### 3.1 Validate the Ontology

```bash
hindsight validate-ontology --ontology data/<domain>/ontology.yaml
```

This checks:
- YAML syntax and schema compliance
- Unique concept IDs
- Valid `confusable_with` references
- At least one concept per family

### 3.2 Generate Model Predictions

Run one or more models on the new domain's documents using the standardized extraction prompt. The prompt should list all canonical concept IDs as expected JSON keys:

```
Extract the following fields from this document. Return a JSON object with these exact keys:

- patient_name
- patient_dob
- date_of_service
- admission_date
- attending_physician_name
...

Return ONLY the JSON object. If a field is not present, use "n/a" as the value.
```

Save predictions in the standard format:

```json
{
  "document_id": "doc_0001",
  "model": "claude-sonnet-4-20250514",
  "predictions": {
    "patient_name": "Jane M. Doe",
    "patient_dob": "03/15/1985",
    "date_of_service": "07/20/2025",
    "admission_date": "07/20/2025"
  }
}
```

### 3.3 Score the Results

```bash
hindsight score \
  --ontology data/<domain>/ontology.yaml \
  --ground-truth data/<domain>/ground_truth/ \
  --predictions results/<domain>/<model_run>.json \
  --output results/<domain>/<model_run>_scores.json
```

This produces:
- Aggregate metrics: field F1, CBA-strict, CBA-soft, CBA-weighted, delta
- Per-document scores
- Misbinding log with concept-level detail
- Confusion matrix (concept x concept)

### 3.4 Analyze Results

At minimum, compute and report:

1. **Delta**: Is there a measurable gap between field F1 and CBA-strict? If delta < 1%, the domain may not exhibit enough concept-binding ambiguity to be informative.

2. **Confusion matrix**: Which concept pairs are most frequently confused? Do the observed confusions align with the `confusable_with` predictions in the ontology?

3. **Ambiguity correlation**: Is there a positive correlation between annotated ambiguity level and observed misbinding rate?

4. **CBA-soft vs. CBA-strict**: Are most misbindings within-family or cross-family?

---

## Step 4: Submit a Pull Request

### 4.1 PR Contents

A complete domain addition PR should include:

```
data/<domain>/
  ontology.yaml                    # Concept ontology
  images/                          # Document images (or instructions for downloading)
  ground_truth/                    # Concept-level annotations
  metadata.json                    # Corpus metadata and statistics

results/<domain>/
  <model_run>_scores.json          # At least one model's scored results

docs/
  # Updates to existing docs if needed (e.g., adding domain-specific
  # guidance to annotation_guidelines.md)
```

### 4.2 PR Description Template

```markdown
## New Domain: <domain_name>

### Summary
- **Domain**: <brief description>
- **Documents**: <count> (<source>)
- **Concepts**: <count> across <family_count> semantic families
- **Predicted confusion pairs**: <list top 3-5>

### Results (at least one model)
| Model | Field F1 | CBA-strict | Delta | Misbindings |
|-------|----------|------------|-------|-------------|
| <model> | <score> | <score> | <score> | <count> |

### Checklist
- [ ] Ontology YAML passes validation
- [ ] >= 50 annotated documents
- [ ] >= 10% double-annotated with kappa >= 0.80
- [ ] At least one model scored with hindsight-eval
- [ ] Datasheet section added (or docs/datasheet.md updated)
- [ ] No PII or proprietary data in submitted documents
```

### 4.3 Review Criteria

Domain addition PRs will be reviewed for:

1. **Ontology quality**: Are definitions precise and unambiguous? Are confusable pairs well-motivated?
2. **Annotation quality**: Is inter-annotator agreement acceptable? Are ambiguity annotations present?
3. **Corpus diversity**: Sufficient layout and noise variation for the domain?
4. **Baseline results**: Does at least one model produce a non-zero delta, indicating that concept-binding ambiguity exists in this domain?
5. **Licensing and privacy**: Are the source documents publicly available or properly licensed? Is there any PII that needs to be redacted?

---

## Appendix: Supported Document Types

The following domains are currently included or planned for the Hindsight benchmark:

| Domain | Status | Concepts | Documents | Primary Source |
|--------|--------|----------|-----------|---------------|
| Paystubs | Active | 27 | 200 | Synthetic (template engine) |
| Invoices/Receipts | In progress | ~20 | TBD | SROIE, FATURA |
| Contracts | In progress | ~18 | TBD | CUAD |
| W-2 Tax Forms | In progress | ~15 | TBD | HuggingFace synthetic |

Potential future domains (contributions welcome):

| Domain | Motivation | Expected Ambiguity |
|--------|-----------|-------------------|
| Medical records (EHR) | Multiple date types, provider vs. patient identity, diagnosis vs. procedure codes | High |
| Insurance claims | Claim amount vs. approved amount vs. paid amount vs. deductible; multiple date types | High |
| Bank statements | Transaction types, balance vs. available balance, debit vs. credit amounts | Moderate |
| Academic transcripts | Course grades vs. GPA, credit hours attempted vs. earned, term vs. cumulative | Moderate |
| Government forms (1099, I-9) | Structured layout like W-2 but with more entity types | Low--Moderate |
