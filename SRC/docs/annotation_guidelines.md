# Hindsight: Annotation Guidelines for Concept-Level Labeling

## 1. Purpose

This document provides standardized guidelines for annotating documents in the Hindsight benchmark with **concept-level labels**. Unlike traditional field-level annotation (which labels extracted values with surface-level field names such as "amount" or "date"), concept-level annotation assigns each value to a **canonical concept ID** drawn from a domain-specific ontology.

Concept-level annotation is the foundation of the Concept-Binding Accuracy (CBA) metric. The quality of CBA scores depends entirely on the quality and consistency of these annotations.

---

## 2. Key Terminology

| Term | Definition |
|------|------------|
| **Canonical concept ID** | A unique, machine-readable identifier for a semantic concept (e.g., `gross_pay_current`, `agreement_date`). Defined in the domain ontology YAML file. |
| **Surface label** | The text that appears on the document itself to identify a field (e.g., "Gross Pay", "Total Earnings", "Compensation"). Multiple surface labels may map to the same canonical concept. |
| **Semantic family** | A group of related concepts (e.g., the "compensation" family includes `gross_pay_current`, `net_pay_current`, `regular_pay_current`, etc.). Used for CBA-soft scoring. |
| **Confusable pair** | Two concepts that are structurally, visually, or semantically similar enough that a reasonable annotator or model might assign a value to either. Listed in the `confusable_with` field of each ontology entry. |
| **Misbinding** | An error where the correct value is extracted but assigned to the wrong canonical concept. |
| **Ambiguity level** | A 0--3 rating of how difficult it is to determine the correct concept assignment for a given field instance (see Section 5). |

---

## 3. How to Assign Canonical Concept IDs

### 3.1 General Procedure

For each value on a document:

1. **Read the ontology definition** for the domain. Every canonical concept has a precise natural-language definition in the ontology YAML file. The definition is the authoritative reference, not the surface label.

2. **Identify the value** on the document. Note its surface label (if any), its position on the page, and any contextual cues (adjacent labels, column headers, section headings).

3. **Match to the best-fitting concept**. Compare the value and its context against all ontology definitions. Assign the concept whose definition most precisely describes the semantic role of that value.

4. **Record the assignment** as a `{concept_id: value}` pair in the annotation JSON.

5. **If ambiguous, assign the most likely concept** and record the ambiguity (see Section 4).

### 3.2 Decision Rules

When the correct concept is not immediately obvious, apply these rules in order:

1. **Definition over surface label**: If the document says "Total Pay" but the value represents gross earnings before deductions, assign `gross_pay_current`, not `net_pay_current`, regardless of the label.

2. **Specificity preference**: When a value could match both a general and a specific concept, prefer the more specific one. For example, if a value is specifically the federal income tax amount, assign `federal_tax_withheld` rather than `total_deductions_current`.

3. **Temporal scope**: Distinguish current-period from year-to-date values. A value labeled "Total" in a YTD column is `gross_pay_ytd`, not `gross_pay_current`.

4. **Directional clarity**: For concepts that differ by direction (e.g., employer vs. employee address, licensor vs. licensee), use positional cues, section headers, and entity references to determine directionality.

5. **Structural context**: When values lack explicit labels, use position on the page (e.g., "first address block is typically employer, second is employee") and mathematical relationships (e.g., "this value equals gross minus deductions, so it must be net pay").

### 3.3 Values Not in the Ontology

If a value on the document does not correspond to any concept in the ontology:

- **Do not annotate it.** Only annotate values that map to a defined canonical concept.
- **Do not invent new concept IDs** without updating the ontology through the formal process (see `docs/adding_domains.md`).
- **Note it in the annotation log** if the omitted value is potentially important or confusable with an existing concept.

---

## 4. Handling Ambiguous Cases

Ambiguity arises when a value could reasonably be assigned to two or more canonical concepts. This is not an annotation failure --- it is the phenomenon that CBA is designed to measure.

### 4.1 When a Value Could Be Multiple Concepts

For each ambiguous field, record:

1. **Primary assignment**: The concept you judge to be most likely correct. This goes into the ground-truth annotation.

2. **Alternative concept(s)**: The other concept(s) that a reasonable annotator or model might choose. These go into the ambiguity annotation.

3. **Disambiguation cue**: What information you used to resolve the ambiguity. Categories of cues:
   - **Positional**: Location on the page (e.g., "appears in the employer section header")
   - **Label-based**: Explicit text label adjacent to the value (e.g., "labeled 'Fed W/H'")
   - **Mathematical**: Arithmetic relationship to other values (e.g., "equals 6.2% of gross pay, confirming Social Security tax")
   - **Domain knowledge**: Understanding of document conventions (e.g., "W-2 Box 1 is always wages/tips/compensation")
   - **Contextual**: Surrounding values or document structure (e.g., "appears in the same table row as 'FICA'")

4. **Ambiguity level**: A 0--3 rating (see Section 5).

### 4.2 Annotation Format for Ambiguous Fields

```json
{
  "document_id": "ps_0061",
  "ground_truth": {
    "employer_address": "147 birch court",
    "employee_address": "347 main circuit"
  },
  "ambiguity": {
    "employer_address": {
      "level": 2,
      "alternatives": ["employee_address"],
      "cue": "positional --- employer name appears directly above this address block",
      "notes": "Heavy scan noise obscures section divider between employer and employee areas"
    }
  }
}
```

---

## 5. Ambiguity Level Scale

Rate each annotated field on a 0--3 scale reflecting how difficult it is to determine the correct concept assignment.

| Level | Label | Definition | Example |
|-------|-------|------------|---------|
| **0** | Unambiguous | No reasonable alternative concept exists. The value's identity is self-evident from its label, position, or format. | Employee name on a paystub (only one person's name, clearly labeled "Employee") |
| **1** | Weakly ambiguous | An alternative concept exists in theory, but a careful reader would resolve it without difficulty using available cues. | Pay date vs. pay period end when they differ by several days and have distinct labels |
| **2** | Moderately ambiguous | Requires domain knowledge, spatial reasoning, or mathematical verification to resolve. A non-expert might choose incorrectly. | State income tax vs. Social Security tax when they appear in adjacent rows with similar dollar amounts and abbreviated labels ("ST TAX" vs. "SS TAX") |
| **3** | Highly ambiguous | Even domain experts might disagree, or resolution requires information not present on the document. | Subtotal vs. total on a receipt where both labels appear but the relationship between them is unclear; agreement date vs. effective date when the contract uses "as of" language ambiguously |

### Calibration Examples by Domain

#### Paystubs

| Value | Level | Reasoning |
|-------|-------|-----------|
| Employee full name in labeled "Employee" section | 0 | Unambiguous: unique identifier, clearly labeled |
| Pay date when labeled "Check Date" and different from period end | 1 | Weakly ambiguous: "Check Date" is a synonym for pay_date; clear if you know the convention |
| Social Security tax ($310.00) next to Medicare tax ($72.50) in a "Taxes" section with no individual labels | 2 | Moderately ambiguous: requires knowing the ~4.3:1 ratio between SS and Medicare rates to resolve |
| Gross pay current ($5,000) when gross pay YTD ($5,000) has the same value (first pay period of the year) | 2 | Moderately ambiguous: identical values; must use positional context ("Current" vs. "YTD" column headers) |

#### Invoices / Receipts

| Value | Level | Reasoning |
|-------|-------|-----------|
| Store name in large font at top of receipt | 0 | Unambiguous: prominent position, distinctive formatting |
| Dollar amount labeled "TOTAL" at bottom of receipt | 1 | Weakly ambiguous: could be pre-tax subtotal or post-tax total, but "TOTAL" at the bottom conventionally means final amount |
| Dollar amount in a row below subtotal but without a clear label | 2 | Moderately ambiguous: could be tax amount, total, or a discount line |
| Amount labeled "TOTAL" when a "GRAND TOTAL" also exists on the same receipt | 3 | Highly ambiguous: "TOTAL" could be subtotal or a category total, and without layout analysis, the distinction is unclear |

#### Contracts

| Value | Level | Reasoning |
|-------|-------|-----------|
| "This Agreement is entered into as of January 15, 2024" | 1 | Weakly ambiguous: "as of" could indicate agreement_date or effective_date, but most readers would classify as agreement_date |
| "Party A shall not, during the term, engage in any business that competes with Party B" | 2 | Moderately ambiguous: could be non_compete or exclusivity; the distinction requires legal expertise |
| "Neither party shall solicit any person who is or was an employee or customer of the other party" | 3 | Highly ambiguous: combines no_solicit_customers and no_solicit_employees in a single clause; annotators must decide whether to assign one concept or split |

#### W-2 Tax Forms

| Value | Level | Reasoning |
|-------|-------|-----------|
| Value in Box 1 of a standard W-2 | 0 | Unambiguous: Box 1 is always wages_tips_compensation by IRS definition |
| Value in Box 3 (Social Security wages) vs. Box 5 (Medicare wages) when both boxes contain the same amount | 1 | Weakly ambiguous: identical values, but box numbers resolve the concept unambiguously |
| Value in Box 12a when the code is partially illegible | 2 | Moderately ambiguous: Box 12 codes (D, DD, W, etc.) determine the concept, and without a readable code, the value could represent multiple benefit types |

---

## 6. Per-Domain Annotation Guidance

### 6.1 Paystubs

**Ontology file**: `data/paystubs/ontology.yaml` (27 concepts, 5 families)

**Key considerations**:

- **Surface label variation**: Paystub templates use diverse labels for the same concept. "Gross Pay", "Total Earnings", "Gross Compensation", and "Total Gross" all map to `gross_pay_current`. Always refer to the ontology definition, not the surface label.

- **Current vs. YTD**: Most paystubs display both current-period and year-to-date columns. Ensure you assign the correct temporal scope. If a value appears in a "YTD" column, it maps to the `*_ytd` variant of the concept.

- **Tax deduction hierarchy**: Annotate specific tax types (`federal_tax_withheld`, `state_tax_withheld`, `social_security_tax`, `medicare_tax`) rather than the aggregate `total_deductions_current`, unless only the aggregate is present.

- **Addresses**: Paystubs often have two address blocks (employer and employee). Use the section header or proximity to the entity name to determine which is which. If a paystub has only one address, annotate it with the appropriate concept and do not fabricate the missing address.

- **Missing fields**: If a concept is not present on the document (e.g., no overtime pay for a salaried employee), do not include it in the ground-truth annotation. The absence of a field is not a misbinding.

### 6.2 Invoices / Receipts

**Ontology file**: `data/invoices/ontology.yaml` (to be created; ~20 concepts)

**Key considerations**:

- **SROIE re-annotation**: When re-annotating SROIE documents, do not blindly adopt the original 4-field labels. Each SROIE "total" must be re-examined to determine whether it represents a subtotal, tax-inclusive total, or amount tendered.

- **Multiple entities**: Receipts may list a franchise name, a parent company, and a branch identifier. Assign each to the most specific applicable concept.

- **Date ambiguity**: A single date on a receipt is typically the transaction date. If multiple dates appear (e.g., a "print date" in a footer), annotate each with the correct concept.

- **Currency and formatting**: Normalize currency values by removing symbols and thousands separators. Record values as plain decimal strings (e.g., "1234.56", not "$1,234.56").

### 6.3 Contracts

**Ontology file**: `data/contracts/ontology.yaml` (to be created; ~18 concepts)

**Key considerations**:

- **Clause-level annotation**: Unlike paystubs and receipts (which have discrete field-value pairs), contracts require annotating text spans. The value is the relevant clause text, and the concept is the clause type.

- **Multi-concept clauses**: A single sentence may contain multiple concepts (e.g., a clause that addresses both non-compete and exclusivity). In such cases, annotate the dominant concept and record alternatives in the ambiguity annotation.

- **Legal expert validation**: All contract annotations must be reviewed by a domain expert. Annotations by non-lawyers should be treated as preliminary and flagged for review.

- **CUAD re-mapping**: When using CUAD annotations, map the 41-category labels to the reduced ~18-concept ontology. Some CUAD categories will merge (e.g., CUAD's "Affiliate License-Licensor" and "Affiliate License-Licensee" may map to a single `affiliate_license` concept with a direction attribute).

### 6.4 W-2 Tax Forms

**Ontology file**: `data/w2/ontology.yaml` (to be created; ~15 concepts)

**Key considerations**:

- **Box numbers as ground truth**: W-2 forms have a standardized layout defined by the IRS. Box numbers are the definitive disambiguation cue. A value in Box 2 is always `federal_tax_withheld`, regardless of any other context.

- **When box numbers are illegible**: If scan noise obscures the box number, the annotator must use positional reasoning (relative to other boxes) and mathematical relationships (e.g., Box 4 should be approximately 6.2% of Box 3) to determine the correct concept.

- **State section**: Boxes 15--20 may contain data for one or two states. Annotate as `state_1_*` and `state_2_*` concepts as applicable.

- **Box 12 codes**: Box 12 values depend on the letter code (D = 401k, DD = employer-sponsored health coverage cost, W = HSA, etc.). If the code is present, use it to select the specific concept. If the code is missing or illegible, annotate with the generic `box_12_value` concept and set ambiguity level to 2+.

---

## 7. Quality Control

### 7.1 Double Annotation

A randomly selected 10% subset of documents in each domain must be independently annotated by two annotators. This serves two purposes:

1. **Inter-annotator agreement measurement**: Reported as Cohen's kappa on concept assignment. Target: kappa >= 0.85.
2. **Ambiguity validation**: Disagreements between annotators are not errors --- they are empirical evidence of concept-binding ambiguity. Disagreements should be recorded and analyzed.

### 7.2 Disagreement Resolution Protocol

When two annotators assign different concept IDs to the same value:

1. **Record both annotations** in the disagreement log with each annotator's reasoning.
2. **Classify the disagreement**:
   - **Annotator error**: One annotator misread the document or misunderstood the ontology definition. Resolution: correct the error.
   - **Genuine ambiguity**: Both assignments are defensible given the available cues. Resolution: assign the majority concept as ground truth, record the alternative in the ambiguity annotation, and set the ambiguity level to >= 2.
   - **Ontology gap**: The ontology definitions do not adequately distinguish the two concepts. Resolution: refine the ontology definitions and re-annotate affected documents.
3. **A third annotator (or domain expert) adjudicates** if the two primary annotators cannot reach consensus.

### 7.3 Consistency Checks

Automated checks should be run on all annotations before they are finalized:

| Check | Description |
|-------|-------------|
| **Schema validation** | Every concept ID in the annotation exists in the domain ontology YAML |
| **Completeness** | No required concepts are missing (domain-specific required concept lists) |
| **Mathematical consistency** | Where applicable, verify arithmetic relationships (e.g., gross_pay - total_deductions = net_pay, within rounding tolerance) |
| **Duplicate detection** | No two fields in the same document annotation have the same concept ID (unless the ontology explicitly allows multiples) |
| **Value format** | Values conform to expected formats (dates as MM/DD/YYYY or similar, currency as decimal strings) |

### 7.4 Annotation Metadata

Each annotated document must include:

```json
{
  "document_id": "ps_0061",
  "domain": "paystub",
  "annotator_id": "annotator_01",
  "annotation_date": "2025-06-15",
  "annotation_version": 1,
  "review_status": "reviewed",
  "reviewer_id": "annotator_02",
  "ground_truth": { ... },
  "ambiguity": { ... },
  "notes": "Heavy scan noise makes employer/employee address blocks hard to distinguish"
}
```

---

## 8. Annotation Workflow Summary

```
1. Load document image and domain ontology YAML
                    |
2. For each identifiable value on the document:
   a. Read the value and note its surface label, position, and context
   b. Match to the best-fitting canonical concept (refer to ontology definitions)
   c. Record the {concept_id: value} pair
   d. If ambiguous, record alternatives and disambiguation cue
   e. Assign ambiguity level (0-3)
                    |
3. Run automated consistency checks
                    |
4. Submit for double-annotation (if in the 10% QC subset)
                    |
5. Resolve any disagreements via the disagreement protocol
                    |
6. Finalize annotation with metadata
```

---

## 9. Versioning

This is a living document. As ontologies are refined and new domains are added, these guidelines will be updated. The annotation version number in each document's metadata tracks which version of the guidelines was used.

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-06 | Initial guidelines for paystub domain |
| 1.1 | 2025-07 | Added invoice/receipt and W-2 guidance |
| 1.2 | TBD | Added contract domain guidance after legal expert review |
