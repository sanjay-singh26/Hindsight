# CBA for Financial Contracts: Guide for Legal Domain Expert

## What This Document Is

This is a guide for you — our legal domain expert — to help set up a research experiment. We're testing whether AI models make a specific type of mistake when extracting information from contracts: **extracting the right text but filing it under the wrong category**.

We call this **Concept Binding Accuracy (CBA)**. You don't need to code anything. You need to apply your legal expertise to three specific tasks described below.

The engineering side (running models, scoring results) will be handled separately.

---

## Background: The Problem We're Measuring

When AI reads a contract and extracts clauses, current evaluation methods check: *"Did the model find the right text?"* (Field-level accuracy).

They don't check: *"Did the model understand what kind of clause it found?"*

For example, a model might correctly extract the sentence *"Party A shall not, for a period of 2 years, solicit any customer of Party B"* — but label it as a **Non-Compete** instead of a **No-Solicit of Customers**. Field-level accuracy says "correct." CBA catches the error.

In our paystub experiments, this misbinding gap was small (~1-2%) because paystubs are simple. Contracts should show a much larger gap because clauses are structurally similar and legally distinct.

---

## The CUAD Dataset

**CUAD** (Contract Understanding Atticus Dataset) is a publicly available dataset created by The Atticus Project. It contains:

- **510 commercial contracts** (NDAs, licensing agreements, service agreements, etc.)
- **13,000+ expert annotations** across **41 clause categories**
- Contracts are sourced from SEC EDGAR filings (publicly available)
- Annotations were done by law students and reviewed by attorneys

**Download**: https://huggingface.co/datasets/theatticusproject/cuad-qa

**Paper**: https://github.com/TheAtticusProject/cuad

### The 41 CUAD Categories

| # | Category | # | Category |
|---|----------|---|----------|
| 1 | Document Name | 22 | Minimum Commitment |
| 2 | Parties | 23 | Volume Restriction |
| 3 | Agreement Date | 24 | IP Ownership Assignment |
| 4 | Effective Date | 25 | Joint IP Ownership |
| 5 | Expiration Date | 26 | License Grant |
| 6 | Renewal Term | 27 | Non-Transferable License |
| 7 | Notice Period to Terminate Renewal | 28 | Affiliate License-Licensor |
| 8 | Governing Law | 29 | Affiliate License-Licensee |
| 9 | Most Favored Nation | 30 | Unlimited/All-You-Can-Eat License |
| 10 | Non-Compete | 31 | Irrevocable or Perpetual License |
| 11 | Exclusivity | 32 | Source Code Escrow |
| 12 | No-Solicit of Customers | 33 | Post-Termination Services |
| 13 | Competitive Restriction Exception | 34 | Audit Rights |
| 14 | No-Solicit of Employees | 35 | Uncapped Liability |
| 15 | Non-Disparagement | 36 | Cap on Liability |
| 16 | Termination for Convenience | 37 | Liquidated Damages |
| 17 | ROFR/ROFO/ROFN | 38 | Warranty Duration |
| 18 | Change of Control | 39 | Insurance |
| 19 | Anti-Assignment | 40 | Covenant Not to Sue |
| 20 | Revenue/Profit Sharing | 41 | Third Party Beneficiary |
| 21 | Price Restrictions | | |

---

## Your Three Tasks

### Task 1: Select 15-20 Canonical Concepts

**Goal**: From the 41 CUAD categories, select 15-20 that are most relevant and most likely to be confused by AI extraction.

**How to think about it**: Not all 41 categories are equally important or equally confusable. Some are straightforward (Document Name, Governing Law) and a model won't mix them up. Others are structurally and linguistically similar — those are what we care about.

**What to deliver**: A list of 15-20 categories, grouped into families. Here's a starting proposal — please revise based on your experience:

```
PARTIES & IDENTIFICATION
  - primary_party (who is "Party A" / the company)
  - counterparty (who is "Party B" / the other side)

DATES & DURATION
  - agreement_date (when the contract was signed/dated)
  - effective_date (when terms become active)
  - expiration_date (when the contract ends)
  - renewal_term (auto-renewal period)
  - notice_period_to_terminate (how much notice to end renewal)

RESTRICTIVE COVENANTS
  - non_compete
  - exclusivity
  - no_solicit_customers
  - no_solicit_employees
  - non_disparagement

IP & LICENSING
  - ip_ownership_assignment
  - license_grant
  - license_transferability (non-transferable or not)

LIABILITY & REMEDIES
  - liability_cap
  - liquidated_damages
  - insurance_requirement

TERMINATION & CONTROL
  - termination_for_convenience
  - change_of_control_trigger
```

**Questions for you to answer**:
- Are there categories I'm missing that you frequently see confused in practice?
- Are there categories above that you'd remove (too obvious, never confused)?
- Would you add any categories from your practice that aren't in CUAD's 41?

---

### Task 2: Identify 5 High-Confusion Concept Pairs

**Goal**: Name 3-5 specific pairs of concepts that, in your legal practice, are genuinely hard to distinguish — even for junior associates.

**What makes a "high-confusion pair"**: Two clause types where:
- The language is structurally similar
- They often appear near each other in contracts
- Mixing them up has real legal consequences

**Starting proposals** (please confirm, revise, or replace):

| Pair | My Hypothesis | Your Assessment? |
|------|---------------|-----------------|
| Agreement Date vs Effective Date | Often the same page, sometimes same value. Models conflate "dated as of" with "effective as of" | _[confirm/revise/replace]_ |
| Non-Compete vs Exclusivity | Both restrict competitive activity, similar language, but different legal implications | _[confirm/revise/replace]_ |
| No-Solicit of Customers vs No-Solicit of Employees | Identical clause structure, differ by one word | _[confirm/revise/replace]_ |
| Affiliate License-Licensor vs Affiliate License-Licensee | Same concept, opposite direction (who grants vs who receives) | _[confirm/revise/replace]_ |
| Uncapped Liability vs Cap on Liability | Same section, opposite meanings. One says "no limit", other specifies the limit | _[confirm/revise/replace]_ |

**For each pair you confirm/add, please also note**:
1. **Severity**: If an AI mixed these up during deal review, how bad is it? (Critical / Significant / Minor)
2. **Real-world example**: A brief sentence on when this confusion would matter. E.g., *"Confusing non-compete with exclusivity matters because non-competes are unenforceable in California, so missing one in a CA deal means missing a void clause."*

---

### Task 3: Validate 10 Contracts from CUAD

**Goal**: Pick 10 contracts from the CUAD dataset that are diverse and representative of real practice.

**Selection criteria**:
- Mix of contract types (NDA, licensing, service, partnership, etc.)
- Mix of complexity (short/simple to long/complex)
- Include at least 2-3 contracts where your selected high-confusion pairs are likely to appear
- Avoid contracts that are too boilerplate/template-like (we want real ambiguity)

**What to deliver**: A list of 10 contract filenames from CUAD with a brief note on why each was chosen.

**Optional but valuable**: For each contract, skim the existing CUAD annotations and flag any you disagree with. E.g., *"CUAD labels this clause as Non-Compete but I'd call it Exclusivity because..."* — these disagreements are exactly the kind of ambiguity CBA is designed to measure.

---

## What Happens After Your Input

Once you deliver the three tasks above, here's what happens on the engineering side:

1. **We build an extraction prompt** using your 15-20 canonical concepts
2. **We send each of the 10 contracts** to Claude and GPT models with that prompt
3. **We score the outputs** against CUAD ground truth (adjusted by your corrections) using CBA
4. **We measure the delta** between field-level accuracy and concept-binding accuracy
5. **We analyze which pairs** the models confuse most — and compare against your predicted high-confusion pairs

Your expertise turns a generic NLP benchmark into a legally meaningful evaluation. The paper's argument is: *"AI extraction looks accurate by standard metrics, but systematically misclassifies clauses in ways that matter to practitioners."* That "in ways that matter" part can only come from you.

---

## Timeline

| Task | Estimated Time | Deadline |
|------|---------------|----------|
| Task 1: Select 15-20 concepts | 2-3 hours | _[to be set]_ |
| Task 2: Identify 5 confusion pairs | 1-2 hours | _[to be set]_ |
| Task 3: Pick & review 10 contracts | 3-4 hours | _[to be set]_ |
| Engineering: Run experiments | 1-2 days (after your input) | — |
| Joint: Review results | 1-2 hours together | — |

---

## Questions?

If anything above is unclear or you'd approach this differently based on your practice, let's discuss. The framework is flexible — your legal judgment should drive the concept selection, not the other way around.
