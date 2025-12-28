# Updates to tfidf_analysis.py

## Summary of Changes

This document summarizes the updates made to `tfidf_analysis.py` to align topic modeling outputs with the supplementary table naming scheme and reorganize the output directory structure.

## Changes Made

### 1. Output Directory Renamed
- **Old:** `OUTPUT_DIR = 'results'`
- **New:** `OUTPUT_DIR = 'semantic_results'`
- **Location:** Line 72

All analysis outputs (figures, tables, and summary) will now be saved to `semantic_results/` instead of `results/`.

### 2. Topic Modeling Tables Renamed to ST3 and ST4

#### ST3: Topic Metadata (formerly `topic_scatter_summary.csv`)
- **Old filename:** `topic_scatter_summary.csv`
- **New filename:** `ST3.csv`
- **Location:** Line 748
- **Description:** Contains topic modeling metadata derived from LDA analysis, including:
  - Topic identifier
  - 2D coordinates for visualization (derived from Jensen-Shannon divergence)
  - Topic prevalence (proportion of corpus assigned to each topic)
  - Top 5 most representative terms for each topic

#### ST4: Document-Topic Distribution (formerly `topic_document_distribution.csv`)
- **Old filename:** `topic_document_distribution.csv`
- **New filename:** `ST4.csv`
- **Location:** Line 749
- **Description:** Document-topic probability distribution matrix where:
  - Each row corresponds to one article from ST2
  - Columns represent probability of article belonging to each of 8 topics
  - Additional 'source' column categorizes articles as PubMed, Preprints, or Other

### 3. Enhanced Documentation

#### Updated Module Docstring (Lines 1-20)
Added explicit mention of topic modeling outputs:
```python
   d. Topic modeling → LDA analysis producing ST3 (topic metadata) and ST4 (doc-topic matrix)

OUTPUTS:
- semantic_results/tables/ST3.csv: Topic modeling metadata
- semantic_results/tables/ST4.csv: Document-topic probability distribution
```

#### Added Inline Comments (Lines 745-749)
```python
# Save as Supplementary Tables ST3 and ST4
# ST3: Topic metadata with coordinates, prevalence, and top terms
# ST4: Document-topic probability distribution matrix
```

#### Updated Summary Section (Lines 762-792)
Added comprehensive descriptions of ST1-ST4 that align with paper manuscript format:

**Supplementary Table 3 (ST3)** contains topic modeling metadata derived from Latent Dirichlet Allocation (LDA) analysis on the curated dataset. Each row represents one of 8 topics identified through unsupervised topic modeling, including: topic identifier, 2D coordinates for visualization (derived from Jensen-Shannon divergence), topic prevalence (proportion of corpus assigned to this topic), and the top 5 most representative terms for each topic. This table supports the topic modeling scatter plot (Supplementary Figure 4) and provides insight into the semantic structure of the included literature.

**Supplementary Table 4 (ST4)** presents the document-topic probability distribution matrix from the LDA analysis. Each row corresponds to one article from ST2, with columns representing the probability that the article belongs to each of the 8 identified topics. An additional 'source' column categorizes articles as PubMed, Preprints, or Other. This table enables analysis of topic distribution patterns across different publication venues and supports reproducibility of the topic modeling results.

#### Updated Output Messages (Lines 926-927)
```python
print(f"    - ST3.csv: Topic modeling metadata (coordinates, prevalence, top terms)")
print(f"    - ST4.csv: Document-topic probability distribution matrix")
```

## Alignment with Manuscript

These changes ensure consistency with the supplementary tables as described in the manuscript:

- **ST1**: Cleaned dataset after deduplication and initial triage
- **ST2**: Extended dataset with manual annotations and semantic tags
- **ST3**: Topic modeling metadata (NEW - this update)
- **ST4**: Document-topic probability distribution (NEW - this update)

All descriptions now match the format used for ST1 and ST2 in the paper, providing clear explanations of:
- What the table contains
- How the data was generated
- What columns are included
- How the table supports the analysis

## Migration Notes

If you have existing results in the `results/` directory, you can:

1. Rename the directory: `mv results semantic_results`
2. Or run the script fresh to generate new outputs in `semantic_results/`
3. The old `topic_scatter_summary.csv` and `topic_document_distribution.csv` files (if they exist) can be deleted after verifying the new ST3.csv and ST4.csv are generated correctly

## Next Steps

To generate the updated tables:

```bash
cd /home/toharhymes/work/otta_bioinf/llm_genetics/llm_in_diagnostics/tf_idf
python tfidf_analysis.py
```

This will create:
- `semantic_results/tables/ST3.csv`
- `semantic_results/tables/ST4.csv`
- All other analysis outputs in `semantic_results/`
