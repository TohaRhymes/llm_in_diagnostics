# Analysis Scripts

This directory contains all analysis scripts and notebooks for the systematic review.

## Pipeline Overview

The analysis follows a sequential pipeline:

```
0. Data Fetching → 1. Merge → 2. Clean & Filter → 3. Manual Review → 4-6. Analysis
```

## Main Scripts (Run in Order)

### Data Collection
- **0_fetch_ncbi.py** - Fetch articles from PubMed
- **0_fetch_arxiv.py** - Fetch articles from arXiv
- **0_fetch_biomed.py** - Fetch articles from bioRxiv/medRxiv
- **utils_fetch.py** - Shared utilities for fetching

### Data Processing
- **1_merge_all.ipynb** - Merge data from all sources
- **2_clean_EDA_and_clinical_extraction.ipynb** - Clean, deduplicate, and filter data
  - Output: `data/processed/cleaned.csv` (51,613 articles)
  - Output: `data/processed/clinic_genetic.csv` (576 articles)

### Analysis (After Manual Annotation)
- **4_st2_analysis.ipynb** - Analyze ST2 (manually annotated data)
  - Output: `data/final/coded_st2.csv`
  
- **5_tfidf_analysis.py** - TF-IDF analysis and topic modeling
  - Output: `figures/SFig1-5` (supplementary figures)
  - Output: `results/tfidf/tables/` (all CSV tables)
  - Output: `results/tfidf/docs/ANALYSIS_SUMMARY.md`
  
- **6_citation_analysis.py** - Citation usage analysis
  - Output: `figures/fig3.pdf` (usage histogram)
  - Output: `results/citations/` (tables and statistics)

## Helper Tools

- **3_highlight.py** - Terminal tool for reviewing abstracts with keyword highlighting

## Usage

### Quick Regeneration of All Figures

```bash
cd scripts

# Regenerate ST2 analysis
jupyter nbconvert --to python --execute 4_st2_analysis.ipynb

# Regenerate TF-IDF figures (SFig1-5)
python3 5_tfidf_analysis.py

# Regenerate citation analysis (fig3)
python3 6_citation_analysis.py
```

### Data Fetching (Optional)

Only needed if updating the dataset:

```bash
python3 0_fetch_ncbi.py
python3 0_fetch_arxiv.py
python3 0_fetch_biomed.py
```

## Dependencies

See main README.md for full dependency list.

## Notes

- Manual annotation is done on `data/final/ST2.csv`
- Do not modify `data/raw/` or `data/processed/` directly
- All figures are automatically saved to `figures/`
- Analysis results go to `results/`
