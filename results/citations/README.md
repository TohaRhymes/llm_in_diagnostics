# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a citation analysis tool for the systematic review project "A Systematic Review on the Generative AI Applications in Human Medical Genomics" (Changalidis et al., 2025, arXiv:2508.20275). The tool analyzes which articles from the manually annotated supplementary table (ST2) are cited in specific sections of the LaTeX manuscript, and visualizes the comparison between total available articles and those actually used in the review.

## Core Script

**`analyze_citations.py`** - Main analysis script that:
1. Parses a LaTeX manuscript to extract citations from specific sections
2. Matches citations against articles in ST2_v2.csv based on their categorization
3. Generates a summary table comparing total articles vs. cited articles
4. Creates a horizontal grouped bar chart visualization

## Running the Analysis

```bash
python analyze_citations.py
```

The script requires:
- Input: `data/ST2_v2.csv` (annotated article database)
- Input: `data/templateArxiv.tex` (LaTeX manuscript)
- Output: `results_table.csv` (summary statistics)
- Output: `results_plot.pdf` (visualization)
- Output: `detailed_citations.csv` (per-article citation details)

## Data Structure

**ST2_v2.csv columns**:
- `ref` - Citation key matching LaTeX \cite{} references
- `final_category` - Main article category (e.g., 'KN', 'CDA', 'GDA', 'COM', 'intro', 'areas')
- `subcategory` - Subcategory code (e.g., 'NER/RE', 'CDP', 'GVI')
- Multiple categories/subcategories separated by commas indicate cross-referenced articles

**Category Mapping** (defined in `SECTION_MAPPING`):
- `intro` → Introduction section
- `KN` → Knowledge Navigation (subcats: NER/RE, RD)
- `CDA` → Clinical Data Analysis (subcats: CDN, CDP, MDP, COP)
- `GDA` → Genetic Data Analysis (subcats: AVE, GVI, PP)
- `COM` → Communication (subcats: MPC, PC)
- `areas` → Related Research Areas
- `techniques` → LLMs Selection Guide & Model Strategies (Discussion)
- `data` → Data and Benchmarks (Discussion)
- `biases` → Biases (Discussion)

## Key Implementation Details

**LaTeX Parsing**:
- Extracts text between section markers using regex patterns
- Handles citations in formats: `\cite{key}` or `\cite{key1, key2, ...}`
- Section boundaries defined by `latex_section` and `end_section` patterns in `SECTION_MAPPING`

**Citation Matching**:
- Filters citations to only include those present in ST2_v2.csv
- Tracks which sections cite each article
- Handles articles with multiple category assignments

**Visualization**:
- Horizontal grouped bar chart with two bars per category
- Blue bars: "Articles after reviewing" (total in ST2)
- Orange bars: "Articles cited in manuscript" (actually used)
- Categories grouped and separated by horizontal lines
- Section labels positioned on the right side

## Dependencies

- pandas - data manipulation
- matplotlib, seaborn - visualization
- numpy - numerical operations
- re, collections - text parsing and data structures

## Task Context

This tool was created to answer the question: "Of all the articles we manually reviewed and categorized, how many did we actually cite in each section of our manuscript?" This provides insight into the coverage and utilization of the systematic review dataset.

## Parent Project

This directory is part of the larger systematic review project located in the parent directory (`../`), which includes:
- Data fetching scripts (0_fetch_*.py)
- Data cleaning and analysis notebooks
- Manual annotation results (ST1.csv, ST2.csv)
- Main project documentation in `../CLAUDE.md`
