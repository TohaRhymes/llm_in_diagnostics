# TF-IDF Analysis for Systematic Review

## Quick Start

```bash
python tfidf_analysis.py
```

**Configuration**: Set `TOP_N = 30` at the top of the script to change the number of phrases analyzed.

Results will appear in the `results/` folder:
- **`figures_main/`** - Main figures for the paper (PDF format)
  - Fig3: Source comparison after selection + filtering (PubMed vs Preprints, 52 union terms)
- **`figures_supplement/`** - Supplementary figures (PDF format)
  - SFig1: 3-panel progression (Full → Selected → Selected+Filtered)
  - SFig2: Source comparison after selection, before filtering (PubMed vs Preprints, 43 union terms)
  - SFig3: Supplement copy of filtered comparison (mirrors Fig3)
  - SFig4: Topic modeling scatter (LDA topics with prevalence bubbles)
- **`tables/`** - CSV tables with top-30 phrases
  - Standard, filtered, and fine-tuned variants (ALL, PubMed, Preprints) + topic-model outputs
- **`ANALYSIS_SUMMARY.md`** - Brief summary of results
- **`FIGURE_CAPTIONS.md`** - Complete figure captions for the paper

**Note**: Comparison plots show the UNION of top-N from both sources, so they may display more than N terms (e.g., if overlap is 8/30, union is 52 terms).

## What the Script Does

### Pipeline:

**Part 1: Three-panel Progression Figure (SFig1)**

Three horizontal bar chart panels stacked vertically showing analytical progression:
1. **Panel A**: Full Dataset (51,613 articles) - broad research landscape before curation
2. **Panel B**: Selected Articles (187 articles) - curated dataset with all terms (generic + specific)
3. **Panel C**: Selected + Filtered (187 articles) - specific research trends after removing 206 generic AI/ML terms

- Purpose: Demonstrate systematic refinement from comprehensive coverage to specific insights
- Output: `SFig1_progression_top30.pdf` (supplement)
- Format: Horizontal bars for readability, panel labels A/B/C at top-left without boxes

**Part 2: Source Comparison Figures**

Compare PubMed (n=125) vs Preprints (n=62) using vertical grouped bar charts:
- **SFig2**: After selection, before filtering → 57% overlap (17/30) → 43 union terms
  - Shows consensus on core topics (validates systematic coverage)
- **Fig3**: After selection + filtering → 27% overlap (8/30) → 52 union terms
  - Shows complementary contributions (exposes source-specific trends)

- All use grouped bars on same scale with enlarged text for readability
- Display actual TF-IDF scores (never 0 for missing terms)
- **All analyses standardized to top-30** (configurable via TOP_N)
- **Union can be 30-60 terms** depending on overlap

**Part 3: Reviewer-Motivated Additions (New)**

- **Fine-Tuned TF-IDF (tables/finetuned\_*.csv):** Train on curated data with full context, then mask AI/ML-native anchors post-hoc and renormalize per document. Confirms that context-preserving masking surfaces the same domain-specific terms; values are shipped via CSV + `ANALYSIS_SUMMARY.md`.
- **SFig3:** Supplement copy of the filtered source comparison (same data as Fig3) per reviewer request for explicit Supplement coverage.
- **SFig4 (Topic Scatter):** Eight-topic LDA fitted on the curated set, projected via PCA. Bubble size encodes prevalence; annotations show the top five bigram terms so overlap/adjacency is obvious.

These additions satisfy Reviewer #2’s TF-IDF variant request and provide the requested topic-model scatter without overloading the main text.

### Results Structure:

```
results/
├── figures_main/              # Main figures FOR PAPER (PDF)
│   └── Fig3_selected_filtered_comparison_top30.pdf  ← Source comparison (selected+filtered, 52 union terms)
│
├── figures_supplement/        # Supplementary figures FOR SUPPLEMENT (PDF)
│   ├── SFig1_progression_top30.pdf                  ← 3-panel progression (Full → Selected → Selected+Filtered)
│   ├── SFig2_selected_comparison_top30.pdf          ← Source comparison (selected, before filtering, 43 union terms)
│   ├── SFig3_selected_filtered_comparison_top30.pdf ← Supplement copy of filtered comparison
│   └── SFig4_topic_scatter.pdf                      ← LDA scatter (prevalence bubbles)
│
├── tables/                    # Data in CSV format (all top-30)
│   ├── full_dataset_top30.csv
│   ├── standard_ALL_top30.csv
│   ├── standard_PubMed_top30.csv
│   ├── standard_Preprints_top30.csv
│   ├── filtered_ALL_top30.csv
│   ├── filtered_PubMed_top30.csv
│   ├── filtered_Preprints_top30.csv
│   ├── finetuned_ALL_top30.csv
│   ├── finetuned_PubMed_top30.csv
│   ├── finetuned_Preprints_top30.csv
│   ├── topic_scatter_summary.csv
│   ├── topic_document_distribution.csv
│   └── source_comparison_top30.csv              ← Overlap statistics (standard, filtered, fine-tuned)
│
├── ANALYSIS_SUMMARY.md        # Brief summary
└── FIGURE_CAPTIONS.md         # Complete figure captions for the paper
```

**Note**: Filenames include TOP_N value (currently 30). Change TOP_N in the script to generate different values.

## For the Paper

### Main Text

**Figure:**
- **Figure 2**: `figures_main/Fig3_selected_filtered_comparison_top30.pdf`
  - Source comparison after selection and filtering (52 union terms from top-30 each)
  - Shows 27% overlap (8/30), demonstrating complementary source contributions
  - See `FIGURE_CAPTIONS.md` for complete caption

**Methods Text (~250 words):**
> We performed TF-IDF (Term Frequency-Inverse Document Frequency) analysis to characterize research themes across three levels: the full literature dataset (51,613 articles), our selected/curated review dataset (187 articles), and a filtered version removing generic terms. Analysis used bigrams and trigrams (ngram_range=(2,3)) with scikit-learn's TfidfVectorizer, extracting up to 1,000 features (max_features=1000) to balance comprehensiveness with computational efficiency. We removed English stop words plus custom terms including citation artifacts ("et al") and standardized all analyses to top-30 phrases for consistency.
>
> The full dataset analysis established baseline coverage of the LLM-genomics landscape. Selected articles (curated through manual review of 187 papers) demonstrated systematic capture of core literature, showing expected dominance of terms like "large language models," "deep learning," and "precision medicine." To reveal specific research trends beyond obvious keywords, we applied a filtered analysis removing 206 generic AI/ML terms (e.g., "language model," "deep learning," "artificial intelligence," "machine learning," "natural language processing"), retaining domain-specific concepts.
>
> We stratified analysis by publication source (PubMed n=125, preprints n=62), using vertical grouped bar charts for direct comparison on the same scale. Overlap was calculated as the number of shared phrases in the top-30 from each source. Comparison plots display the UNION of top-30 from both sources (43-52 unique terms depending on overlap), showing actual TF-IDF scores for all terms including those not in both top-30 lists, avoiding artificial zeros and providing accurate relative importance across sources.

**Results Text (~250 words):**
> TF-IDF analysis revealed a systematic progression from generic to specific research themes (Supplementary Figure S1). Full dataset analysis (51,613 articles, Panel A) showed expected dominance of "language models" (TF-IDF: 0.0200), "large language" (0.0200), and "artificial intelligence" (0.0174), validating comprehensive field coverage before curation. Selected articles (187 papers, Panel B) maintained this pattern with "large language" (TF-IDF: 0.0463), "language models" (0.0441), and "artificial intelligence" (0.0244) dominating, confirming systematic capture of core literature. After filtering 206 generic AI/ML terms (Panel C), specific research trends emerged: 'precision medicine' (TF-IDF: 0.0175), 'gene expression' (0.0160), 'pre trained' (0.0154), 'open source' (0.0138), and 'genetic testing' (0.0116) topped the rankings. Clinical applications emphasized 'electronic health' records, disease-specific research featured 'breast cancer' and 'alzheimer disease', knowledge resources included 'human phenotype ontology', and emerging techniques like 'attention mechanism' and 'single cell' genomics appeared prominently.
>
> Source comparison before filtering (Supplementary Figure S2) showed 57% overlap (17/30 phrases) between PubMed (n=125) and preprints (n=62), with 43 union terms, indicating strong consensus on core topics. After filtering (Figure 2), overlap dropped to 27% (8/30 phrases) with 52 union terms, revealing complementary source-specific contributions. PubMed emphasized clinical validation ('precision medicine', TF-IDF 0.0175; 'genetic testing', 0.0116) and established medical concepts ('breast cancer', 'alzheimer disease', 'clinical genetics'), while preprints focused on emerging computational methods ('gene expression', 0.0160; 'open source', 0.0138; 'intelligence ai', 0.0123) and novel techniques ('attention mechanism', 'fine tuned'). This complementary coverage confirms the value of including both publication types in systematic reviews of rapidly evolving fields.

### Supplement

**Figures:**
- **Supplementary Figure S1**: `figures_supplement/SFig1_progression_top30.pdf`
  - Three-panel progression (Full → Selected → Selected+Filtered)
  - Shows systematic refinement from 51,613 articles to specific trends
- **Supplementary Figure S2**: `figures_supplement/SFig2_selected_comparison_top30.pdf`
  - Source comparison after selection, before filtering
  - Shows 57% overlap (17/30), validating systematic coverage
- **Supplementary Figure S3**: `figures_supplement/SFig3_selected_filtered_comparison_top30.pdf`
  - Supplement copy of Fig3 (filtered comparison) for completeness
- **Supplementary Figure S4**: `figures_supplement/SFig4_topic_scatter.pdf`
  - Topic-model scatter (LDA, PCA projection with prevalence bubbles)

**Complete captions available in `FIGURE_CAPTIONS.md`**

**Short Captions:**
> **Supplementary Figure S1:** Progression of TF-IDF Analysis from Full Dataset to Filtered Insights. Three panels showing (A) Full dataset (51,613 articles) with generic terms dominant, (B) Selected articles (187) maintaining core topics, (C) Selected + Filtered revealing specific trends after removing 206 generic AI/ML terms. Horizontal bars show top-30 phrases in each stage.
>
> **Supplementary Figure S2:** Source Comparison After Article Selection. PubMed (n=125) vs preprints (n=62) before filtering generic terms. 57% overlap (17/30 phrases) indicates strong consensus on core topics. Plot displays 43 union terms with actual TF-IDF scores on same scale.
>
> **Supplementary Figure S4:** Topic scatter derived from an eight-topic LDA model fitted on the curated articles. PCA-based coordinates show relative proximity; bubble sizes encode topic prevalence, and annotations list the top five bigram terms for each topic.

### Reviewer-Focused Additions (draft text)

> **Exploratory TF-IDF usage.** TF-IDF serves as a supporting semantic screen that complements, but does not drive, the manuscript’s final conclusions. We now ship four reproducible variants: (i) the full-dataset baseline, (ii) the curated review set with all terminology intact, (iii) the filtered view (main text), and (iv) the context-preserving “fine-tuned” view (`tables/finetuned_*`) where we train with full context and mask AI/ML anchors post-hoc. Because the fine-tuned model is always trained with full context, removing the anchors after scoring does not change downstream findings; it simply highlights the next-most informative genomic phrases. We summarize its top terms in `ANALYSIS_SUMMARY.md` so reviewers can see the stability without another figure.
>
> **Topic-model scatter request.** Added Supplementary Figure S4 (`figures_supplement/SFig4_topic_scatter.pdf`), which plots eight LDA topics with PCA coordinates and prevalence based bubble sizes. This covers the reviewer’s scatter suggestion while keeping the figure count manageable.

## Interpretation

### Why Two Datasets?

**Full Dataset (cleaned.csv):**
- Shows we understand the complete literature landscape
- For context: "Here's what exists in the field"
- Validates our understanding before curation

**Curated Dataset (ST2.csv):**
- Shows what we actually included in the review
- Demonstrates systematic selection
- Enables focused analysis

### Why These Variants?

**Standard (supplement):**
- Shows we covered all major topics
- For reviewers: "Yes, we systematically covered LLMs + genomics"
- Validates review structure

**Filtered (main text):**
- Shows specifics and diversity
- For readers: "Here are the specific things researchers are doing"
- Demonstrates field maturity

**Fine-Tuned (tables/finetuned\_*.csv):**
- Implements Reviewer #2’s “fourth variant”: train with full context, then mask AI/ML anchors post-hoc and renormalize.
- Confirms that removing generic phrases after fitting keeps rankings stable while highlighting the same domain-specific concepts.

**Topic Scatter (SFig4):**
- Visualizes eight LDA topics as a scatter plot with prevalence-based bubble sizes.
- Shows topic overlap/adjacency and satisfies the scatter-plot request without adding more bar charts.

### Source Comparison (Grouped Bars)

- Direct visual comparison on same scale (top-30 phrases from each source)
- **After selection, before filtering (SFig2)**: 57% overlap (17/30) → 43 union terms
  - Shows consensus on core topics
  - Validates systematic coverage of both source types
- **After selection + filtering (Fig3)**: 27% overlap (8/30) → 52 union terms
  - Shows complementary source-specific contributions
  - Exposes unique perspectives after removing generic terms

**Key Methodological Features:**
- Comparison plots show UNION, not just overlap
  - If overlap = 30/30 (100%), union = 30 terms
  - If overlap = 0/30 (0%), union = 60 terms
  - Standard case: 17/30 overlap → 30 + 30 - 17 = 43 union terms
  - Filtered case: 8/30 overlap → 30 + 30 - 8 = 52 union terms
- All terms show actual TF-IDF scores (never 0 for missing terms)
- Increased text size (11-14pt) and figure dimensions for readability
- Output format: PDF for publication quality

**Conclusion:** PubMed + Preprints = complementary coverage, justifying inclusion of both source types

### Hyperparameters

**TF-IDF Configuration:**
- **ngram_range=(2, 3):** Captures bigrams and trigrams for multi-word phrases (e.g., "large language models", "precision medicine")
- **max_features=1000:** Balances comprehensiveness with computational efficiency; sufficient to capture major trends
- **stop_words:** English stop words + custom ones (citation artifacts: "et al", etc.)
- **TOP_N=30:** Standardized across all analyses for consistency and readability (configurable parameter)

**Comparison Plot Methodology:**
- Takes union of top-N from both sources (can be 30-60 terms depending on overlap)
- Displays actual TF-IDF score for each term in both sources
- Terms not in top-N of one source still show their real (often low) TF-IDF score
- This provides accurate comparison without artificial zeros
- Plot height auto-adjusts based on number of union terms

## Dependencies

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Input Data

- `data/cleaned.csv` - Full deduplicated dataset (51,613 articles)
- `data/ST2.csv` - Curated dataset (187 articles after filtering by 'what section used' column)

## Reproducibility

- **Random seed:** Not used (TF-IDF is deterministic)
- **Runtime:** ~2-3 minutes (includes processing 51,613 articles)
- **Output:**
  - 1 main figure (Fig3) + 2 supplement figures (SFig1, SFig2) - PDF format, 300 DPI
  - 7 CSV tables (full + standard + filtered, all top-30)
  - 1 overlap statistics CSV
  - Figure captions document
- **Consistency:** All analyses use top-30 phrases for comparability
- **Configuration:** Edit `TOP_N = 30` at top of script to change N (filenames auto-update)

---

**Created:** 2025-10-18
**Updated:** 2025-10-20
**Status:** ✅ Final version

**Major Changes:**
- **PDF output:** Changed from PNG to PDF for publication quality
- **Renamed files:** More descriptive names reflecting content
  - `SFig1_progression` (Full → Selected → Selected+Filtered)
  - `SFig2_selected_comparison` (before filtering)
  - `Fig3_selected_filtered_comparison` (after filtering)
- **3-panel figure:** Combined full/selected/filtered into SFig1
- **Panel labels:** Removed boxes, positioned at top-left (-0.08, 1.05)
- **Readability:** Increased font sizes (labels 11pt, title 14pt, axis 13pt)
- **Terminology:** "Selected" (curated) vs "Filtered" (generic terms removed)
- **Precise statistics:**
  - 51,613 articles (full), 187 selected (125 PubMed, 62 preprints)
  - 206 generic terms filtered
  - 57% overlap before filtering (17/30 → 43 union)
  - 27% overlap after filtering (8/30 → 52 union)
- **Figure captions:** Complete captions in `FIGURE_CAPTIONS.md`
