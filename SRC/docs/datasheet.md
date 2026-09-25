# Datasheet for the Hindsight Benchmark Dataset

Following the framework proposed by Gebru et al. (2021), "Datasheets for Datasets," *Communications of the ACM*, 64(12), 86--92.

---

## 1. Motivation

### 1.1 For what purpose was the dataset created?

The Hindsight benchmark dataset was created to evaluate **Concept-Binding Accuracy (CBA)** in document intelligence systems. Existing document AI benchmarks (e.g., SROIE, FUNSD, DocVQA) measure field-level extraction accuracy --- whether a model finds the correct value --- but do not measure whether the model assigns that value to the correct semantic concept. The Hindsight dataset provides concept-level annotations that enable evaluation of this distinction.

The dataset supports the empirical claim that standard field-level metrics overstate model performance by failing to detect *misbindings*: cases where the correct value is extracted but assigned to the wrong concept (e.g., extracting "$4,523.67" correctly but labeling it as "Net Pay" instead of "Gross Pay").

### 1.2 Who created the dataset and on behalf of which entity?

The dataset was created by the authors as part of the Hindsight research project.

### 1.3 Who funded the creation of the dataset?

The dataset creation was self-funded by the research team. Estimated total cost for API calls during model evaluation is $85--165. Dataset construction itself incurred no monetary cost beyond researcher labor, as all source documents are either synthetically generated or drawn from publicly available datasets.

### 1.4 Any other comments?

The dataset is designed to be a living benchmark. New domains can be added by following the process described in `docs/adding_domains.md`. The initial release covers four document domains; the framework is intended to generalize to any document type with structured information extraction tasks.

---

## 2. Composition

### 2.1 What do the instances that comprise the dataset represent?

Each instance represents a single document (rendered as an image) paired with concept-level ground-truth annotations. The annotations map extracted values to canonical concept identifiers defined in domain-specific ontologies.

The dataset spans four document domains:

| Domain | Instance Type | Approximate Count |
|--------|--------------|-------------------|
| Paystubs | Synthetic paystub images | 200 |
| Invoices/Receipts | Receipt images from SROIE and FATURA | 100--150 (planned) |
| Contracts | Legal contract text from CUAD | 50--80 (planned) |
| W-2 Tax Forms | Synthetic W-2 form images | 50--100 (planned) |

### 2.2 How many instances are there in total?

The planned total is 300--530 instances across all four domains. As of the current release, approximately 200 instances (paystubs) are fully annotated.

### 2.3 Does the dataset contain all possible instances or is it a sample?

The dataset is a sample. For each domain:

- **Paystubs**: A synthetic sample generated from 15--20 template layouts with randomized field values. The population of possible paystub configurations is effectively infinite.
- **Invoices/Receipts**: A subset of the SROIE dataset (which itself contains 973 receipt images from the ICDAR 2019 competition) and FATURA, re-annotated at the concept level.
- **Contracts**: A subset of the CUAD dataset (which contains 510 contracts with 13,000+ annotations), selected for diversity of contract types and presence of predicted high-confusion clause pairs.
- **W-2 Tax Forms**: A subset of the singhsays/fake-w2-us-tax-form-dataset from HuggingFace, which contains synthetically generated W-2 form images.

### 2.4 What data does each instance consist of?

Each instance consists of:

1. **Document image**: A PNG or JPEG file containing the rendered document
2. **Ground-truth annotation (JSON)**: A mapping of canonical concept IDs to their correct values for that document
3. **Ambiguity annotation (JSON)**: For each annotated field, an ambiguity level (0--3) and, for ambiguous fields, a list of alternative concept IDs with disambiguation cues
4. **Metadata**: Domain, layout type, noise type, annotator ID, annotation date, and version

### 2.5 Is there a label or target associated with each instance?

Yes. The ground-truth annotation provides the label for each instance. Labels are structured as `{concept_id: value}` pairs, where `concept_id` is drawn from the domain ontology and `value` is the string representation of the extracted information.

### 2.6 Is any information missing from individual instances?

Not all canonical concepts are present in every document. For example, a paystub for a salaried employee may not include `overtime_pay_current`. Missing concepts are intentionally omitted from the ground-truth annotation (they are not annotated as "n/a"). This is by design: the absence of a concept is meaningful and distinct from a concept being present but unextracted.

### 2.7 Are relationships between individual instances made explicit?

No explicit relationships exist between instances. Documents are independent. However, within the paystub domain, documents share template layouts (the `layout` metadata field), and within the invoice/receipt and contract domains, documents may come from the same source dataset with shared characteristics.

### 2.8 Are there recommended data splits?

Not yet formalized. For model evaluation, we recommend stratified random splits by layout type and noise level to ensure that evaluation subsets are representative. A suggested split for the paystub domain:

| Split | Purpose | Size |
|-------|---------|------|
| Development | Prompt engineering and debugging | 20 documents |
| Validation | Hyperparameter selection | 30 documents |
| Test | Final reported results | 150 documents |

### 2.9 Are there any errors, sources of noise, or redundancies?

- **Synthetic generation artifacts**: Paystub and W-2 documents are synthetically generated. While templates are designed to mimic real-world documents, they may lack certain idiosyncrasies found in actual payroll systems (e.g., custom employer addenda, handwritten corrections).
- **Intentional noise**: A subset of paystub documents includes simulated degradation (scan artifacts labeled `heavy_scan`, phone camera distortion labeled `phone_photo`). This noise is intentional and documented in the metadata.
- **OCR-inherited errors**: For SROIE-sourced receipts, ground-truth values are derived from OCR output and may contain transcription errors from the original dataset. These are corrected where identified during concept-level re-annotation.

### 2.10 Is the dataset self-contained, or does it link to or otherwise rely on external resources?

The dataset is largely self-contained. Document images and annotations are included in the repository. However:

- **SROIE and CUAD source data** must be downloaded separately from their respective repositories due to licensing considerations. Download instructions are provided.
- **The HuggingFace W-2 dataset** must be downloaded separately.
- **Domain ontology YAML files** are included in the repository and are required for scoring.

### 2.11 Does the dataset contain data that might be considered confidential?

No. All data is either synthetically generated or derived from publicly available datasets:

- **Paystubs**: Entirely synthetic. Employee names, employers, addresses, and financial figures are randomly generated and do not correspond to real individuals or organizations.
- **Invoices/Receipts**: From SROIE (ICDAR 2019 public competition data) and FATURA (public dataset). These contain real merchant names and addresses from publicly issued receipts.
- **Contracts**: From CUAD, which sources contracts from SEC EDGAR filings (publicly available government records).
- **W-2 Forms**: Synthetic, from a HuggingFace dataset of fake W-2 forms. No real SSNs or EINs are included.

### 2.12 Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No. The documents are financial and legal records with no offensive content.

### 2.13 Does the dataset relate to people?

The synthetic documents (paystubs, W-2 forms) contain fictitious personal information (names, addresses, SSN fragments). These are randomly generated and do not correspond to real individuals. The SROIE receipts contain real merchant information but no personal data. CUAD contracts contain names of real companies but are sourced from public SEC filings.

---

## 3. Collection Process

### 3.1 How was the data associated with each instance acquired?

Data acquisition varies by domain:

- **Paystubs**: Generated programmatically using a custom template engine (`paystub_generator/`). The engine produces parameterized HTML/CSS paystub layouts, renders them to images, and outputs ground-truth JSON annotations. Randomization covers employee demographics, pay amounts, tax calculations, addresses, and layout selection.

- **Invoices/Receipts**: Obtained from the publicly available SROIE dataset (ICDAR 2019 Robust Reading Competition, Task 3) and FATURA dataset. Original field-level annotations (4 fields: company, date, address, total) are re-annotated at the concept level using the expanded Hindsight invoice ontology.

- **Contracts**: Obtained from the CUAD dataset (The Atticus Project). Original 41-category clause annotations are re-mapped to the reduced Hindsight contract ontology (~18 concepts). A legal domain expert validates the mappings and flags disagreements.

- **W-2 Forms**: Obtained from the singhsays/fake-w2-us-tax-form-dataset on HuggingFace. Synthetic W-2 images are annotated with canonical concept IDs derived from the IRS box numbering system.

### 3.2 What mechanisms or procedures were used to collect the data?

- **Synthetic generation**: Python scripts with Jinja2 templates, Pillow/wkhtmltoimage for rendering, and NumPy for randomization
- **Public dataset download**: HuggingFace `datasets` library and direct HTTP download from SROIE/CUAD repositories
- **Re-annotation**: Manual annotation by trained annotators following the procedures in `docs/annotation_guidelines.md`
- **Expert review**: Legal domain expert review for the contract domain

### 3.3 If the dataset is a sample from a larger set, what was the sampling strategy?

- **SROIE**: Stratified random sample to ensure diversity of receipt types (grocery, restaurant, retail, service)
- **CUAD**: Purposive sampling guided by the legal domain expert, selecting contracts that contain predicted high-confusion clause pairs and span diverse contract types (NDA, licensing, service, partnership)
- **W-2**: Random sample from the HuggingFace dataset, with additional noise variants applied to a subset

### 3.4 Who was involved in the data collection process?

- **Paystub generation**: Research team (software engineers)
- **SROIE re-annotation**: Research team annotators, following standardized guidelines
- **CUAD selection and validation**: Legal domain expert (external collaborator)
- **W-2 annotation**: Research team annotators
- **Quality control**: Double-annotation by a second annotator on a 10% subset across all domains

### 3.5 Over what timeframe was the data collected?

Data collection and annotation began in Week 1 of the research project (approximately May 2025) and is expected to continue through Week 4 (approximately June 2025).

### 3.6 Were any ethical review processes conducted?

No formal IRB review was conducted, as the dataset does not involve human subjects. All personal information in the dataset is either fictitious (synthetic documents) or publicly available (SEC filings, public competition data).

### 3.7 Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources?

All data is obtained from public sources or generated synthetically. No data was collected directly from individuals.

---

## 4. Preprocessing / Cleaning / Labeling

### 4.1 Was any preprocessing/cleaning/labeling of the data done?

Yes:

- **Image preprocessing**: Synthetic documents are rendered at standardized resolutions. Noise variants are generated by applying simulated scan artifacts (gaussian blur, speckle noise, rotation) and phone camera effects (perspective distortion, uneven lighting) using OpenCV.

- **Value normalization**: Ground-truth values are normalized to consistent formats:
  - Currency: Plain decimal strings without symbols or commas (e.g., "4523.67")
  - Dates: Preserved in the format present on the document (e.g., "01/15/2025" or "January 15, 2025")
  - Names: Preserved as they appear on the document, including middle initials

- **Concept-level labeling**: The primary labeling effort. Each value is assigned a canonical concept ID from the domain ontology. See `docs/annotation_guidelines.md` for the full labeling protocol.

- **Ambiguity annotation**: A secondary labeling pass assigns ambiguity levels (0--3) and records alternative concept assignments for ambiguous fields.

### 4.2 Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?

Yes. For synthetic documents, the generation parameters are saved alongside the rendered images, enabling regeneration. For SROIE and CUAD, the original dataset annotations are preserved; Hindsight concept-level annotations are stored as a separate annotation layer.

### 4.3 Is the software that was used to preprocess/clean/label the data available?

Yes. The paystub template engine is included in the repository (`paystub_generator/`). Image noise generation scripts are included in `scripts/`. Annotation tools are standard JSON editors; no custom annotation software was developed.

---

## 5. Uses

### 5.1 Has the dataset been used for any tasks already?

Yes. The paystub domain has been used for pilot evaluation experiments (Phases 0--3 of the research):

- **Phase 0**: 10 clean paystubs evaluated with Claude Sonnet (delta = 0.0)
- **Phase 1**: 10 adversarial paystubs with confusable layouts (delta = 0.007)
- **Phase 2**: Ablation study with Sonnet and Haiku across prompt variants (delta = 0.0 to 0.022)
- **Phase 3**: 50 image-based paystubs with visual noise (delta = 0.005 to 0.007, 7--9 misbindings)

### 5.2 Is there a repository that links to any or all papers or systems that use the dataset?

The primary paper is in preparation (target: NeurIPS 2026 Datasets and Benchmarks track). The repository at the current location serves as the canonical reference.

### 5.3 What (other) tasks could the dataset be used for?

Beyond CBA evaluation, the dataset could support:

- **Document layout analysis**: The diversity of templates and noise types provides training/evaluation data for layout detection models
- **OCR evaluation**: Comparing OCR accuracy across noise levels on documents with known ground-truth text
- **Prompt engineering research**: Studying how prompt formulation affects structured extraction accuracy
- **Active learning research**: Using ambiguity annotations to study whether models can identify their own uncertainty about concept assignments
- **Synthetic document generation research**: The template engine and its outputs could inform research on synthetic training data for document AI

### 5.4 Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

- **Synthetic bias**: The paystub and W-2 domains use synthetic documents. Models trained or fine-tuned on these documents may not generalize to real-world documents with different formatting conventions, handwritten elements, or idiosyncratic layouts.
- **English-only**: All documents are in English. CBA evaluation on multilingual documents would require new ontologies and annotations.
- **U.S.-centric**: Paystubs and W-2 forms follow U.S. payroll and tax conventions. Paystubs from other countries have different concepts (e.g., no Social Security tax in some jurisdictions).
- **Limited contract diversity**: The CUAD dataset primarily contains U.S. commercial contracts from SEC filings, which may not represent the full diversity of legal agreements globally.

### 5.5 Are there tasks for which the dataset should not be used?

- **Training production document extraction models**: The dataset is designed for evaluation, not training. Its size (300--530 documents) is insufficient for training, and its synthetic nature may introduce distributional biases.
- **Real financial analysis**: The financial figures in synthetic documents are randomly generated and do not reflect real economic data.
- **Legal advice**: Contract annotations are for research evaluation purposes and should not be used as legal guidance.

---

## 6. Distribution

### 6.1 Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?

Yes. The dataset (excluding components with incompatible licenses) will be publicly released alongside the research paper.

### 6.2 How will the dataset be distributed?

- **Primary distribution**: GitHub repository
- **Supplementary distribution**: HuggingFace Datasets (planned)
- **Format**: Document images (PNG/JPEG), annotations (JSON), ontologies (YAML), evaluation code (Python)

### 6.3 When will the dataset be distributed?

The initial release is planned to coincide with the paper submission (target: 2026). Pre-release versions are available in the development repository.

### 6.4 Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The Hindsight-specific components (synthetic documents, ontologies, annotations, evaluation code) are released under the **MIT License**.

Source dataset components are subject to their original licenses:
- **SROIE**: Research use (ICDAR competition terms)
- **CUAD**: CC BY 4.0 (The Atticus Project)
- **HuggingFace W-2 dataset**: License as specified by the dataset author

Users must comply with the source dataset licenses when using those components.

### 6.5 Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

SROIE data is subject to ICDAR competition terms, which generally permit research use. CUAD data is CC BY 4.0. No additional restrictions are known.

### 6.6 Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

---

## 7. Maintenance

### 7.1 Who will be supporting/hosting/maintaining the dataset?

The authors. Contact: [redacted for anonymous review].

### 7.2 How can the owner/curator/manager of the dataset be contacted?

Via the GitHub repository issue tracker. [Contact details redacted for anonymous review.]

### 7.3 Is there an erratum?

Not at this time. Errors discovered after release will be documented in a `CHANGELOG.md` file in the repository and, if material, noted in the dataset's HuggingFace card.

### 7.4 Will the dataset be updated?

Yes. Planned updates include:

- **Domain additions**: New document types contributed by the community via the process in `docs/adding_domains.md`
- **Ontology refinements**: Concept definitions may be updated based on annotation disagreements and expert feedback
- **Additional annotations**: Ambiguity annotations and double-annotation coverage will be expanded
- **Bug fixes**: Errors in ground-truth annotations will be corrected as they are identified

Each update will increment the dataset version number.

### 7.5 If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?

The dataset does not contain real personal data. Synthetic personal information (names, addresses, SSN fragments) is fictitious and not subject to data retention regulations. SROIE receipt data contains real merchant information from publicly issued receipts, which is not subject to individual data protection requirements.

### 7.6 Will older versions of the dataset continue to be supported/hosted/maintained?

Yes. All versions will be tagged in the Git repository. Previous versions remain accessible via Git tags and, if published, via versioned HuggingFace dataset revisions.

### 7.7 If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes. See `docs/adding_domains.md` for the process to add new document domains. Contributions are accepted via pull requests to the GitHub repository. All contributions must include ontology YAML, annotated documents, and at least one model's CBA evaluation results.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daume III, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86--92.
- Huang, Z., Chen, K., He, J., et al. (2019). ICDAR 2019 Competition on Scanned Receipts OCR and Information Extraction (SROIE). *ICDAR 2019*.
- Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. *NeurIPS 2021 Datasets and Benchmarks Track*.
