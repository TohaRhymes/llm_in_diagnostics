# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a systematic review project analyzing the application of Large Language Models (LLMs) in human genetic disease research and diagnosis. The project fetches scientific articles from multiple sources (NCBI PubMed, arXiv, bioRxiv, medRxiv), processes and filters them based on relevance to LLMs and medical genomics, and performs text analysis to identify research trends.

**Published Research**: Changalidis et al. (2025) - "A Systematic Review on the Generative AI Applications in Human Medical Genomics" (arXiv:2508.20275)

## Data Pipeline Architecture

**All scripts are located in `scripts/` directory**. The codebase follows a sequential numbered pipeline:

1. **Data Fetching** (`scripts/0_fetch_*.py`):
   - `0_fetch_ncbi.py` - Fetches from PubMed using Biopython Entrez API
   - `0_fetch_arxiv.py` - Fetches from arXiv using arxiv Python package
   - `0_fetch_biomed.py` - Fetches from bioRxiv and medRxiv using their REST APIs
   - All use query terms defined in `utils_fetch.py` and save to `data/raw/0_*.csv`

2. **Data Merging** (`scripts/1_merge_all.ipynb`):
   - Combines all fetched articles from different sources
   - Removes duplicates
   - Outputs to `data/raw/0_all.csv`

3. **Data Cleaning and Analysis** (`scripts/2_clean_EDA_and_clinical_extraction.ipynb`):
   - Cleans text (removes HTML tags, normalizes formatting)
   - Merges duplicate articles by title/abstract
   - Performs TF-IDF and n-gram analysis
   - Filters articles using multi-stage inclusion criteria (LLM terms + clinical terms + genomic terms)
   - Applies exclusion terms to reduce false positives
   - Outputs to `data/processed/cleaned.csv` and `data/processed/clinic_genetic.csv`
   - Generates figures to `figures/` directory

4. **Manual Annotation**:
   - Articles in `data/processed/clinic_genetic.csv` were manually reviewed and annotated in Google Sheets
   - Results stored in `data/final/ST1.csv` and `data/final/ST2.csv` (Supplementary Tables from the paper)
   - See `data/final/ST_readme.txt` for column descriptions and annotation schema

5. **ST2 Analysis** (`scripts/4_st2_analysis.ipynb`):
   - Analyzes the manually annotated supplementary tables

6. **TF-IDF Analysis** (`scripts/5_tfidf_analysis.py`):
   - Generates TF-IDF analysis and all supplementary figures (SFig1-5)
   - Outputs figures to `figures/` directory
   - Saves tables and documentation to `results/tfidf/`
   - See `results/tfidf/README.md` for detailed documentation

7. **Citation Analysis** (`scripts/6_citation_analysis.py`):
   - Analyzes which articles from ST2 were cited in the manuscript
   - Outputs to `results/citations/`

8. **Utility Script** (`scripts/3_highlight.py`):
   - Terminal-based tool to review abstracts with color-coded keyword highlighting
   - Helps quickly identify relevant articles during manual review

## Key Configuration

**Search Parameters** (`utils_fetch.py`):
- Date range: 2023-01-01 to 2025-01-31
- Two query term groups (must match both):
  - LLM-related terms: 'LLM', 'large language model', 'GPT', 'transformer', 'BERT', 'RAG', etc.
  - Medical genomics terms: 'genomic', 'genetic', 'NGS', 'variant interpretation', 'clinical', etc.

**Data Directory**: All scripts use `DATA_DIR='data'` from `utils_fetch.py`. Scripts run from `scripts/` directory use relative paths (`../data/`)

## Running the Pipeline

**Fetch new articles**:
```bash
cd scripts
python 0_fetch_ncbi.py
python 0_fetch_arxiv.py
python 0_fetch_biomed.py
```

**Merge and process**:
```bash
cd scripts
jupyter notebook 1_merge_all.ipynb
jupyter notebook 2_clean_EDA_and_clinical_extraction.ipynb
```

**Generate TF-IDF analysis and figures**:
```bash
cd scripts
python 5_tfidf_analysis.py
```

**Run citation analysis**:
```bash
cd scripts
python 6_citation_analysis.py
```

**Review abstracts with highlighting**:
```bash
cd scripts
python 3_highlight.py
```

## Dependencies

Uses standard scientific Python stack:
- pandas - data manipulation
- Biopython (Bio.Entrez) - NCBI PubMed queries
- arxiv - arXiv API client
- requests - bioRxiv/medRxiv API calls
- sklearn (scikit-learn) - TF-IDF vectorization and text analysis
- matplotlib, seaborn - visualization
- tqdm - progress bars

Note: The conda environment uses Python 3.11.5

## Article Annotation Schema

The supplementary tables (`ST1.csv`, `ST2.csv`) use a structured annotation scheme:

**Diagnostic Phases**:
- `pre:` - Pre-Analytics (Knowledge Navigation, Risk Stratification)
- `ana:` - Analytics (Medical Imaging, Variant Effects, Clinical Variant Interpretation)
- `post:` - Post-Analytics (Patient Clustering, Report Generation, Decision Support)
- `edu` - Educational applications
- `disc` - Discussion/other

**Relevance Scoring**:
- 0 = irrelevant
- 1 = partially relevant
- 2 = fully relevant

See `data/final/ST_readme.txt` for complete field definitions.

## Project Structure

```
llm_in_diagnostics/
├── scripts/                # All code (Python scripts and Jupyter notebooks)
├── data/
│   ├── raw/               # Raw fetched data
│   ├── processed/         # Cleaned and filtered data
│   ├── final/             # Manually annotated supplementary tables
│   └── manuscript/        # LaTeX manuscript files
├── figures/               # All figures (main and supplementary)
├── results/
│   ├── tfidf/            # TF-IDF analysis results and documentation
│   └── citations/        # Citation analysis results
├── deprecated/            # Old versions of files (to be removed)
├── manuscript.pdf         # Published manuscript
├── README.md              # Main project documentation
└── CLAUDE.md              # This file
```

## Important Notes

- **All scripts run from `scripts/` directory** and use relative paths (`../data/`, `../figures/`, etc.)
- The NCBI fetcher uses email "anton@gmail.com" in `Entrez.email` - update this if re-running queries
- Some fetch scripts implement fallback strategies when API limits are hit (e.g., splitting queries)
- Manual deduplication was performed on `clinic_genetic.csv` to produce the final `ST1` table
- **Figures** are saved to `figures/` directory as PDFs
- **Data paths** have been reorganized:
  - Raw data: `data/raw/`
  - Processed data: `data/processed/`
  - Final tables: `data/final/`
- The filtering process uses phrase inclusion with word exclusion to handle false positives (e.g., "tragic" matches "trag" in TF-IDF but isn't relevant)
- **ST2.csv** is the current version (previous references to ST2_v2.csv have been updated)
