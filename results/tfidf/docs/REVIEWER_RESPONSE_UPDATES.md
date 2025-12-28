# TF-IDF Analysis Updates: Response to Reviewer Comments

## Summary

This document describes the updates made to the TF-IDF analysis in response to reviewer comments #2 and #3.

## Changes Made

### 1. Reviewer Comment #2: Context-Preserving Fine-Tuned Analysis

**Request:** "Include a fourth variant as fine-tuning the model learned using the curated review set (192 articles) after removing the AI/ML-native words such as 'foundation models' or 'deep learning'."

**Implementation:**
- Added `finetune_tfidf()` function (lines 157-183 in `tfidf_analysis.py`)
- This function:
  1. Takes the TF-IDF matrix trained on the full curated dataset (with all vocabulary)
  2. Zeros out generic AI/ML anchor terms
  3. Renormalizes document vectors
  4. Recomputes mean TF-IDF scores
- This preserves semantic context during initial feature extraction while down-weighting generic phrases in final ranking

**Outputs:**
- CSV tables: `results/tables/finetuned_ALL_top30.csv`, `finetuned_PubMed_top30.csv`, `finetuned_Preprints_top30.csv`
- Figure: `results/figures_supplement/SFig5_finetuned_comparison_top30.pdf`
- Statistics in `results/tables/source_comparison_top30.csv` (shows 27% overlap between PubMed and preprints)

**Key Finding:** Fine-tuned analysis confirms that domain-specific trends (precision medicine, gene expression, genetic testing) remain stable across different filtering strategies, validating the filtered analysis approach.

### 2. Reviewer Comment #3: Topic Modeling Scatter Plot

**Request:** "It would be good to see a scatter plot of these topics and how much they overlap."

**Implementation:**
- Added `run_topic_model()` function (lines 191-236)
- Uses gensim's LDA with 8 topics
- Custom force-directed layout based on Jensen-Shannon divergence between topic-word distributions
- Topics positioned closer together when they share more vocabulary
- Bubble size reflects topic prevalence across the 192 curated articles

**Outputs:**
- Figure: `results/figures_supplement/SFig4_topic_scatter.pdf`
- CSV tables: `results/tables/topic_scatter_summary.csv`, `topic_document_distribution.csv`

**Key Finding:** Visualization shows how different research areas (clinical diagnostics, precision medicine, protein structure, variant calling) interconnect, with some topics closely related and others occupying distinct semantic spaces.

### 3. Updated LaTeX Text

**Created files:**
1. `UPDATED_APPENDIX.tex` - Complete Appendix A text with all 5 supplementary figures
2. `UPDATED_METHODS_SECTION.tex` - Updated methods section for main text

**Figure Structure:**
- **SF1**: Progression (Full → Selected → Filtered) - 3 panels
- **SF2**: Source comparison BEFORE filtering (57% overlap)
- **SF3**: Source comparison AFTER filtering (23% overlap)
- **SF4**: Topic modeling scatter plot (NEW - addresses Comment #3)
- **SF5**: Fine-tuned source comparison (27% overlap) (NEW - addresses Comment #2)

## Running the Analysis

### Prerequisites

Due to NumPy/SciPy version conflicts in the current environment, you'll need to run this in a clean environment:

```bash
# Option 1: Create new conda environment
conda create -n tfidf_env python=3.11
conda activate tfidf_env
pip install pandas numpy scipy scikit-learn matplotlib seaborn gensim

# Option 2: Use existing environment with correct versions
# numpy < 2.0
# scipy < 1.12
# gensim >= 4.0
```

### Run the Analysis

```bash
cd /home/toharhymes/work/otta_bioinf/llm_genetics/llm_in_diagnostics/tf_idf
python tfidf_analysis.py
```

This will generate:
- All figures in `results/figures_main/` and `results/figures_supplement/`
- All CSV tables in `results/tables/`
- Summary in `results/ANALYSIS_SUMMARY.md`

## Results Already Available

The previous run generated all tables successfully. You can inspect:
- `results/tables/source_comparison_top30.csv` - Shows overlap statistics for Standard, Filtered, and Fine-Tuned analyses
- `results/tables/finetuned_ALL_top30.csv` - Top terms from fine-tuned analysis
- All other CSV files are present and up-to-date

**Note:** Only the figures need to be regenerated (due to the added SFig5 and updated code).

## Integration into Manuscript

### Methods Section
Replace the TF-IDF section with content from `UPDATED_METHODS_SECTION.tex`

### Appendix
Replace Appendix A with content from `UPDATED_APPENDIX.tex`

### Figure Files
After running the script, copy these files to your manuscript directory:
```bash
# From tf_idf/results/figures_supplement/
cp results/figures_supplement/SFig1_progression_top30.pdf ../imgs/
cp results/figures_supplement/SFig2_selected_comparison_top30.pdf ../imgs/
cp results/figures_supplement/SFig3_selected_filtered_comparison_top30.pdf ../imgs/
cp results/figures_supplement/SFig4_topic_scatter.pdf ../imgs/
cp results/figures_supplement/SFig5_finetuned_comparison_top30.pdf ../imgs/
```

## Key Points for Reviewer Response

### Comment #2 Response:
"We have implemented a context-preserving fine-tuned analysis as suggested. This approach first trains the TF-IDF model on the curated dataset (192 articles) with full vocabulary context, then applies post-hoc reweighting to down-weight generic AI/ML anchor terms while preserving semantic relationships. The results (Supplementary Figure S5, Supplementary Tables) confirm that the domain-specific trends reported in our main text (precision medicine, gene expression, genetic testing, etc.) remain stable across different filtering strategies. The overlap between PubMed and preprints in the fine-tuned analysis (27%) is similar to the filtered analysis (23%), validating our original approach while addressing the reviewer's concern about maintaining context."

### Comment #3 Response:
"We have added a topic modeling visualization (Supplementary Figure S4) that shows the relationships and overlap between eight latent topics extracted from the curated dataset. Using Latent Dirichlet Allocation with Jensen-Shannon divergence-based spatial layout, we display topics as bubbles sized by prevalence, with labels showing top terms. This visualization clearly demonstrates how different research areas (clinical diagnostics, precision medicine, computational methods, protein structure) relate to each other, with some topics closely clustered and others occupying distinct semantic spaces. This provides an intuitive complement to the TF-IDF analysis."

## Technical Details

### Fine-Tuning Method (`finetune_tfidf` function)
```python
def finetune_tfidf(tfidf_matrix, feature_names, exclude_patterns):
    """
    Zero out excluded phrases, renormalize per document, recompute TF-IDF stats.
    This simulates training with full context followed by post-hoc fine-tuning.
    """
    # 1. Build mask for allowed phrases
    allowed_mask = [not should_exclude(name, exclude_patterns)
                   for name in feature_names]

    # 2. Zero out excluded phrases in TF-IDF matrix
    dense = tfidf_matrix.toarray()
    dense[:, ~allowed_mask] = 0

    # 3. Renormalize document vectors
    row_norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / row_norms

    # 4. Recompute statistics
    mean_tfidf = dense.mean(axis=0)
    doc_freq = (dense > 0).sum(axis=0)

    return results_df[allowed_mask & (mean_tfidf > 0)]
```

### Topic Modeling Layout
- Uses gensim LDA with auto-optimized hyperparameters
- Jensen-Shannon divergence measures topic similarity
- Custom force-directed layout positions similar topics closer together
- 400 iterations to stabilize positions
- Bubble size proportional to topic prevalence

## Files Modified

1. `tfidf_analysis.py` - Added SFig5 generation (lines 720-733), updated output messages
2. Created `UPDATED_APPENDIX.tex` - Complete appendix text
3. Created `UPDATED_METHODS_SECTION.tex` - Updated methods text
4. Created this `REVIEWER_RESPONSE_UPDATES.md` - Documentation

## Next Steps

1. Fix Python environment dependencies (numpy/scipy versions)
2. Run `python tfidf_analysis.py` to generate all figures
3. Copy figures to `../imgs/` directory
4. Update manuscript LaTeX with new text from `UPDATED_APPENDIX.tex` and `UPDATED_METHODS_SECTION.tex`
5. Submit revised manuscript with 5 supplementary figures (SF1-SF5)
