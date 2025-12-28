# Large Language Models for Genetic Disease Research and Diagnosis

This project presents a systematic review and analysis of how large language models (LLMs) can be applied in the research and diagnosis of human genetic diseases.

**Published Research**: Changalidis et al. (2025) - "A Systematic Review on the Generative AI Applications in Human Medical Genomics" ([arXiv:2508.20275](https://arxiv.org/abs/2508.20275))

## Citation

If you want to cite this research, please use the following format:

```bibtex
@misc{changalidis2025systematicreviewgenerativeai,
      title={A Systematic Review on the Generative AI Applications in Human Medical Genomics},
      author={Anton Changalidis and Yury Barbitoff and Yulia Nasykhova and Andrey Glotov},
      year={2025},
      eprint={2508.20275},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.20275},
}
```

## Project Structure

```
llm_in_diagnostics/
├── scripts/                    # All analysis scripts and notebooks
│   ├── 0_fetch_*.py           # Data fetching from PubMed, arXiv, bioRxiv, medRxiv
│   ├── 1_merge_all.ipynb      # Merge data from all sources
│   ├── 2_clean_EDA_and_clinical_extraction.ipynb  # Data cleaning and filtering
│   ├── 3_highlight.py         # Terminal tool for abstract review
│   ├── 4_st2_analysis.ipynb   # Supplementary table analysis
│   ├── 5_tfidf_analysis.py    # TF-IDF analysis and figure generation
│   ├── 6_citation_analysis.py # Citation usage analysis
│   └── utils_fetch.py         # Shared utilities
│
├── data/                       # All data files
│   ├── raw/                   # Raw data from sources
│   │   ├── 0_ncbi.csv
│   │   ├── 0_arxiv.csv
│   │   ├── 0_bio_med.csv
│   │   └── 0_all.csv          # Combined raw data
│   ├── processed/             # Cleaned and filtered data
│   │   ├── cleaned.csv        # Deduplicated dataset (51,613 articles)
│   │   └── clinic_genetic.csv # Clinically relevant articles (576 articles)
│   ├── final/                 # Final manually annotated tables
│   │   ├── ST1.csv            # Supplementary Table 1
│   │   ├── ST2.csv            # Supplementary Table 2 (manually annotated)
│   │   ├── coded_st2.csv      # Coded version of ST2
│   │   └── ST_readme.txt      # Annotation schema description
│   └── manuscript/
│       └── templateArxiv.tex  # LaTeX manuscript
│
├── figures/                    # All figures for the manuscript
│   ├── fig1.pdf               # Main figure 1
│   ├── fig2.pdf               # Main figure 2 (3-panel histogram)
│   ├── Fig2_a.pdf             # Figure 2 panel A
│   ├── Fig2_b.pdf             # Figure 2 panel B
│   ├── fig4.pdf               # Main figure 4
│   ├── SFig1_progression_top30.pdf           # Supplementary Figure 1
│   ├── SFig2_selected_comparison_top30.pdf   # Supplementary Figure 2
│   ├── SFig3_selected_filtered_comparison_top30.pdf  # Supplementary Figure 3
│   ├── SFig4_topic_scatter.pdf               # Supplementary Figure 4
│   ├── SFig5_finetuned_comparison_top30.pdf  # Supplementary Figure 5
│   ├── workflow_LLM_bio_v2.drawio           # Workflow diagram
│   └── LLM_medical_v6.drawio                # Medical workflow diagram
│
├── results/                    # Analysis results
│   ├── tfidf/                 # TF-IDF analysis results
│   │   ├── tables/            # CSV tables with top-30 phrases
│   │   └── docs/              # Analysis documentation
│   └── citations/             # Citation usage analysis
│       ├── results_table.csv
│       ├── results_plot.pdf
│       └── detailed_citations.csv
│
├── manuscript.pdf              # Published manuscript
├── README.md                   # This file
├── CLAUDE.md                   # Instructions for Claude Code
└── LICENSE
```

## Data Pipeline

The project follows a sequential pipeline for systematic literature review:

### 1. Data Fetching (`scripts/0_fetch_*.py`)
Fetches articles from multiple sources using defined search queries:
- **PubMed** (`0_fetch_ncbi.py`): Uses Biopython Entrez API
- **arXiv** (`0_fetch_arxiv.py`): Uses arxiv Python package
- **bioRxiv/medRxiv** (`0_fetch_biomed.py`): Uses REST APIs

**Search criteria** (defined in `utils_fetch.py`):
- Date range: 2023-01-01 to 2025-01-31
- Two query groups (both must match):
  - LLM-related terms: LLM, GPT, transformer, BERT, RAG, etc.
  - Medical genomics terms: genomic, genetic, NGS, variant interpretation, clinical, etc.

### 2. Data Merging (`scripts/1_merge_all.ipynb`)
Combines all fetched articles and removes duplicates → `data/raw/0_all.csv`

### 3. Data Cleaning and Filtering (`scripts/2_clean_EDA_and_clinical_extraction.ipynb`)
- Cleans text (removes HTML tags, normalizes formatting)
- Merges duplicate articles by title/abstract
- Performs TF-IDF and n-gram analysis
- Multi-stage filtering using inclusion criteria:
  - LLM terms
  - Clinical terms
  - Genomic terms
- Applies exclusion terms to reduce false positives
- **Outputs**:
  - `data/processed/cleaned.csv` (51,613 articles)
  - `data/processed/clinic_genetic.csv` (576 articles)

### 4. Manual Annotation
Articles from `clinic_genetic.csv` were manually reviewed and annotated using defined schema:
- **Diagnostic phases**: Pre-Analytics, Analytics, Post-Analytics, Education
- **Relevance scoring**: 0 (irrelevant), 1 (partially relevant), 2 (fully relevant)
- **Results**: `data/final/ST1.csv` and `data/final/ST2.csv`
- See `data/final/ST_readme.txt` for complete annotation schema

### 5. Analysis Scripts
- **ST2 Analysis** (`scripts/4_st2_analysis.ipynb`): Analyzes manually annotated data
- **TF-IDF Analysis** (`scripts/5_tfidf_analysis.py`): Generates term frequency analysis and figures
- **Citation Analysis** (`scripts/6_citation_analysis.py`): Tracks which articles were cited in manuscript

### 6. Utility Tool (`scripts/3_highlight.py`)
Terminal-based tool to review abstracts with color-coded keyword highlighting

## Running the Pipeline

### Data Fetching
```bash
cd scripts
python 0_fetch_ncbi.py
python 0_fetch_arxiv.py
python 0_fetch_biomed.py
```

### Data Processing
```bash
jupyter notebook 1_merge_all.ipynb
jupyter notebook 2_clean_EDA_and_clinical_extraction.ipynb
```

### Generate TF-IDF Analysis and Figures
```bash
python 5_tfidf_analysis.py
```
**Outputs**: Generates all SFig1-5 PDFs in `figures/` and CSV tables in `results/tfidf/tables/`

### Review Abstracts
```bash
python 3_highlight.py
```

### Citation Analysis
```bash
python 6_citation_analysis.py
```

## Dependencies

```python
# Core
pandas
numpy
matplotlib
seaborn
scikit-learn

# Data fetching
biopython      # PubMed/NCBI queries
arxiv          # arXiv API
requests       # bioRxiv/medRxiv

# Text analysis
gensim         # Topic modeling (LDA)

# Utilities
tqdm           # Progress bars
```

**Environment**: Python 3.11.5 (Conda)

## Key Datasets

- **Full dataset** (`data/processed/cleaned.csv`): 51,613 articles
- **Filtered dataset** (`data/processed/clinic_genetic.csv`): 576 articles
- **Final reviewed** (`data/final/ST2.csv`): Manually annotated articles with detailed classification (see published paper for exact counts)
  - PubMed: 125 articles
  - Preprints (arXiv, bioRxiv, medRxiv): 62 articles

## Documentation

- **TF-IDF Analysis**: See `results/tfidf/README.md` for detailed documentation
- **Citation Analysis**: See `results/citations/README.md`
- **Annotation Schema**: See `data/final/ST_readme.txt`
- **For Claude Code**: See `CLAUDE.md` for development instructions

## License

See `LICENSE` file for details.

## Authors

- Anton Changalidis
- Yury Barbitoff
- Yulia Nasykhova
- Andrey Glotov

## Links

- **Paper**: [arXiv:2508.20275](https://arxiv.org/abs/2508.20275)
- **GitHub**: [Repository](https://github.com/yourusername/llm_in_diagnostics)

---

**Last Updated**: December 2025
