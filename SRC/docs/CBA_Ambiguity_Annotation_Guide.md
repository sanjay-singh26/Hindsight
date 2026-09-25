# CBA Ambiguity Annotation Guide

## Purpose

Before running any model experiments, we need to prove that concept-binding ambiguity **exists in the documents themselves**. This annotation exercise produces the evidence for the paper's core claim.

The output of this exercise becomes **Table 1** in the paper: *"Inter-Concept Ambiguity Analysis Across Document Types."*

---

## How To Annotate

For each document, go field by field and answer three questions:

1. **Could a reasonable person assign this value to a different concept?**
   - YES = genuinely ambiguous
   - NO = unambiguous

2. **If YES, which other concept(s) could it plausibly be?**
   - List the specific alternative concept(s)

3. **What disambiguation cue would a reader use?**
   - Position on page? Label text? Relative value? Domain knowledge?
   - This tells us what a model would need to "understand" to get it right

### Ambiguity Levels

Rate each field:

| Level | Meaning | Example |
|-------|---------|---------|
| **0 — Unambiguous** | No reasonable alternative exists | Employee name on a paystub |
| **1 — Weakly ambiguous** | Alternative exists but a careful reader would resolve it | Pay date vs pay period end (different formats give it away) |
| **2 — Moderately ambiguous** | Requires domain knowledge or spatial reasoning to resolve | State tax vs Social Security tax (similar amounts, adjacent rows) |
| **3 — Highly ambiguous** | Even experts might disagree or need context | Subtotal vs total on a receipt with no clear labels |

---

## Dataset 1: SROIE Receipts (5 documents)

**Source**: SROIE dataset (ICDAR 2019 competition)
**Download**: https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1 or original SROIE

### Concepts to evaluate

SROIE has 4 official fields. But the ambiguity is in what each field *actually means*:

| SROIE Field | Possible Concepts (what could it really be?) |
|-------------|----------------------------------------------|
| company | Store name? Parent company? Franchise name? Brand? |
| date | Transaction date? Print date? Receipt date? |
| address | Store address? HQ address? |
| total | Pre-tax subtotal? Tax-inclusive total? Amount due? Amount tendered? Grand total? |

### Annotation Sheet — SROIE

**For each of 5 receipts, fill in one row per field:**

#### Receipt 1: [filename]

| Field on Receipt | Value | SROIE Label | Ambiguity Level (0-3) | Could Also Be | Disambiguation Cue | Notes |
|-----------------|-------|-------------|----------------------|---------------|-------------------|-------|
| | | company | | | | |
| | | date | | | | |
| | | address | | | | |
| | | total | | | | |

**Also note any additional values on the receipt that are NOT captured by SROIE's 4 fields but could be confused with them:**

| Value on Receipt | What It Actually Is | Could Be Confused With | Why |
|-----------------|--------------------|-----------------------|-----|
| | | | |

*(Repeat for Receipts 2-5)*

---

## Dataset 2: CUAD Contracts (5 documents)

**Source**: CUAD dataset (510 contracts, 41 categories)
**Download**: https://huggingface.co/datasets/theatticusproject/cuad-qa

### High-priority concept pairs to evaluate

Focus on these pairs — for each CUAD annotation, ask whether the labeled span could reasonably belong to the other category:

| Pair | Category A | Category B |
|------|-----------|-----------|
| 1 | Agreement Date (#3) | Effective Date (#4) |
| 2 | Expiration Date (#5) | Notice Period to Terminate Renewal (#7) |
| 3 | Non-Compete (#10) | Exclusivity (#11) |
| 4 | No-Solicit of Customers (#12) | No-Solicit of Employees (#14) |
| 5 | Affiliate License-Licensor (#28) | Affiliate License-Licensee (#29) |
| 6 | Uncapped Liability (#35) | Cap on Liability (#36) |
| 7 | Termination for Convenience (#16) | Notice Period to Terminate Renewal (#7) |

### Annotation Sheet — CUAD

**For each of 5 contracts, review the existing CUAD annotations:**

#### Contract 1: [filename]

| Annotated Span (text) | CUAD Label | Ambiguity Level (0-3) | Could Also Be | Why It's Ambiguous | Disambiguation Cue |
|-----------------------|-----------|----------------------|---------------|-------------------|-------------------|
| | | | | | |
| | | | | | |

**Key question for each annotation**: *"If I showed this clause to another lawyer without the CUAD label, what would they call it?"*

*(Repeat for Contracts 2-5)*

---

## Dataset 3: W-2 Tax Forms (5 documents)

**Source**: singhsays/fake-w2-us-tax-form-dataset (already downloaded)

### Concepts to evaluate

W-2 has a fixed grid layout, so ambiguity should be lower. But test these pairs:

| Pair | Concept A | Concept B | Why Possibly Confusable |
|------|----------|----------|------------------------|
| 1 | wages_tips_compensation | social_security_wages | Both dollar amounts representing wages |
| 2 | social_security_wages | medicare_wages | Adjacent boxes, both wage bases |
| 3 | social_security_tax | medicare_tax | Adjacent boxes, both tax amounts |
| 4 | federal_tax_withheld | social_security_tax | Both tax amounts in same column |
| 5 | employer_address | employee_address | Two address blocks, vertically stacked |
| 6 | employee_ssn | employer_ein | Both ID numbers, adjacent boxes |
| 7 | box_12a_value | box_12b_value | Identical structure, stacked |
| 8 | state_1_wages | state_1_income_tax | Adjacent, both dollar amounts |

### Annotation Sheet — W-2

**Exercise: Cover up the box numbers (a, b, 1, 2, 3...) mentally. For each value, can you still identify which concept it belongs to?**

#### W-2 #1: [filename]

| Value on Form | Correct Concept | Without Box Numbers, Could Also Be | Ambiguity Level (0-3) | What Cue Resolves It |
|--------------|----------------|-----------------------------------|----------------------|---------------------|
| | | | | |
| | | | | |

*(Repeat for W-2s 2-5)*

---

## Dataset 4: Paystubs (5 documents)

**Source**: Your generated paystubs (output/images/)

### Concepts to evaluate

| Pair | Concept A | Concept B | Why Possibly Confusable |
|------|----------|----------|------------------------|
| 1 | gross_pay_current | net_pay_current | Both dollar amounts, sometimes similar |
| 2 | gross_pay_current | gross_pay_ytd | Same concept, different time scope |
| 3 | employer_address | employee_address | Two address blocks |
| 4 | state_tax_withheld | social_security_tax | Similar amounts, adjacent rows |
| 5 | social_security_tax | medicare_tax | Adjacent rows, both FICA |
| 6 | health_insurance | retirement_401k | Adjacent deduction line items |
| 7 | pay_date | pay_period_end | Both dates, sometimes close in value |
| 8 | pay_period_start | pay_period_end | Both dates, definitely close |

### Annotation Sheet — Paystubs

#### Paystub 1: [filename]

| Value on Form | Surface Label on Document | Correct Concept | Ambiguity Level (0-3) | Could Also Be | What Cue Resolves It |
|--------------|--------------------------|----------------|----------------------|---------------|---------------------|
| | | | | | |
| | | | | | |

*(Repeat for Paystubs 2-5)*

---

## Summary Template

After annotating all 20 documents (5 per dataset), fill in this summary:

### Ambiguity Summary by Dataset

| Dataset | Total Fields Reviewed | Level 0 (Unambiguous) | Level 1 (Weak) | Level 2 (Moderate) | Level 3 (High) | % Ambiguous (Level 2+) |
|---------|----------------------|----------------------|----------------|-------------------|----------------|----------------------|
| SROIE Receipts | | | | | | |
| CUAD Contracts | | | | | | |
| W-2 Forms | | | | | | |
| Paystubs | | | | | | |

### Most Common Confusion Pairs (across all datasets)

| Rank | Concept A | Concept B | Dataset | Avg Ambiguity Level | Frequency |
|------|----------|----------|---------|--------------------|-----------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

### Key Finding

*[One paragraph: Does concept-binding ambiguity exist? Where is it strongest? Which document types are most affected? What does this predict for model performance?]*

---

## What Comes Next

This annotation produces:

1. **Table 1 for the paper**: Ambiguity analysis — proof that the problem exists
2. **Hypothesis for experiments**: Which concept pairs should show the highest delta
3. **Go/no-go decision**: If most fields are Level 0-1 across all datasets, CBA may not reveal enough to justify the paper. If many fields are Level 2-3, proceed with confidence.

**After this annotation, THEN run the model experiments.** The experiments test whether models actually make the errors that the ambiguity analysis predicts.
