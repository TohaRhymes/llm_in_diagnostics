#!/usr/bin/env python
# coding: utf-8
"""
TF-IDF Analysis for Systematic Review on LLMs in Medical Genomics

PIPELINE:
1. Full dataset analysis (cleaned.csv 51,613 articles) → overview of field
2. Curated dataset (ST2.csv 187 articles) → what we included in review
   a. Standard analysis → shows core topics (supplement)
   b. Filtered analysis → removes generic terms, shows specific trends (main text)
   c. Source comparison → PubMed vs Preprints (grouped bar plots)

Author: Changalidis Anton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Top N phrases to analyze and display
# Set to 30 for better figure readability
# Note: Comparison plots show UNION of top-N from both sources,
# so they may display more terms (e.g., if no overlap, would show 2*N terms)
TOP_N = 30

# Custom stop words (in addition to English stop words)
# These are domain-specific artifacts that should be filtered out
CUSTOM_STOP_WORDS = [
    'et al',  # Citation artifact
    'et',     # Part of "et al"
    'al',     # Part of "et al"
]

# Output directory
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures_main', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures_supplement', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/tables', exist_ok=True)

# Plot style - increased font sizes by ~20% for better readability
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10  # Base font
plt.rcParams['axes.labelsize'] = 12  # Axis labels: 11→12 pt
plt.rcParams['axes.titlesize'] = 15  # Titles: 12→15 pt
plt.rcParams['xtick.labelsize'] = 10  # Tick labels: 9→10 pt
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11  # Legend: 10→11 pt

print("=" * 80)
print("TF-IDF ANALYSIS: Full Dataset + Curated Dataset Pipeline")
print("=" * 80)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_text(row):
    """Extract title + abstract text"""
    title = str(row.get('title', ''))
    abstract = str(row.get('abstract', ''))
    return f"{title} {abstract}"


def categorize_source(source_str):
    """Categorize source as PubMed, Preprints, or Other"""
    if pd.isna(source_str):
        return 'Other'
    source_lower = str(source_str).lower()
    if 'pubmed' in source_lower:
        return 'PubMed'
    elif any(x in source_lower for x in ['arxiv', 'biorxiv', 'medrxiv']):
        return 'Preprints'
    return 'Other'


def run_tfidf(corpus_df, corpus_name, ngram_range=(2, 3), max_features=1000):
    """
    Run TF-IDF and return results

    Parameters:
        ngram_range (tuple): (2, 3) for bigrams and trigrams - captures multi-word phrases
        max_features (int): 1000 features - balances comprehensiveness with computational efficiency
        stop_words (list): English stop words + custom ones (e.g., "et al")
        lowercase (bool): True - normalizes text for consistent matching
    """
    # Combine English stop words with custom ones
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words_combined = list(ENGLISH_STOP_WORDS) + CUSTOM_STOP_WORDS

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=stop_words_combined,
        lowercase=True
    )

    tfidf_matrix = vectorizer.fit_transform(corpus_df['text'])
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
    doc_freq = np.array((tfidf_matrix > 0).sum(axis=0)).flatten()

    results_df = pd.DataFrame({
        'phrase': feature_names,
        'mean_tfidf': mean_tfidf,
        'doc_freq': doc_freq
    }).sort_values('mean_tfidf', ascending=False)

    print(f"  {corpus_name}: {len(results_df)} phrases extracted")
    return results_df


def should_exclude(phrase, exclude_patterns):
    """Check if phrase should be excluded"""
    phrase_lower = phrase.lower()
    return any(pattern in phrase_lower for pattern in exclude_patterns)


def plot_top_phrases(df, title, filename, n, color='steelblue', ax=None, panel_label=None):
    """Create vertical bar chart for single dataset

    Args:
        df: DataFrame with TF-IDF results
        title: Plot title
        filename: Output filename (only used if ax is None)
        n: Number of top phrases to plot
        color: Bar color
        ax: Matplotlib axes object (if None, creates new figure)
        panel_label: Panel label (e.g., 'A', 'B', 'C') to add in top-left corner
    """
    if ax is None:
        # Standalone figure
        fig_width = max(12, n * 0.35)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        standalone = True
    else:
        standalone = False

    top_n = df.head(n).sort_values('mean_tfidf', ascending=False)

    # Vertical bars with thick black borders
    ax.bar(range(len(top_n)), top_n['mean_tfidf'],
           color=color, alpha=0.8,
           edgecolor='black', linewidth=1)

    ax.set_xticks(range(len(top_n)))
    ax.set_xticklabels(top_n['phrase'], rotation=30, ha='right')
    ax.set_ylabel('Mean TF-IDF Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add panel label if provided
    if panel_label:
        ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')

    if standalone:
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")


def plot_top_phrases_horizontal(df, title, n, color='steelblue', ax=None, panel_label=None):
    """Create horizontal bar chart for single dataset

    Args:
        df: DataFrame with TF-IDF results
        title: Plot title
        n: Number of top phrases to plot
        color: Bar color
        ax: Matplotlib axes object (required for subplots)
        panel_label: Panel label (e.g., 'A', 'B', 'C') to add in top-left corner
    """
    top_n = df.head(n).sort_values('mean_tfidf', ascending=True)  # ascending for horizontal

    # Horizontal bars with thick black borders
    ax.barh(range(len(top_n)), top_n['mean_tfidf'],
           color=color, alpha=0.8,
           edgecolor='black', linewidth=1)

    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n['phrase'], fontsize=9)
    ax.set_xlabel('Mean TF-IDF Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10, fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add panel label if provided
    if panel_label:
        ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')


def plot_three_horizontal_stacked(df1, df2, df3, titles, colors, labels, filename, n):
    """Create figure with 3 horizontal bar charts stacked vertically

    Args:
        df1, df2, df3: DataFrames with TF-IDF results
        titles: List of 3 titles for each subplot
        colors: List of 3 colors for each subplot
        labels: List of 3 panel labels (e.g., ['A', 'B', 'C'])
        filename: Output filename
        n: Number of top phrases to plot in each
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    plot_top_phrases_horizontal(df1, titles[0], n, color=colors[0], ax=axes[0], panel_label=labels[0])
    plot_top_phrases_horizontal(df2, titles[1], n, color=colors[1], ax=axes[1], panel_label=labels[1])
    plot_top_phrases_horizontal(df3, titles[2], n, color=colors[2], ax=axes[2], panel_label=labels[2])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_source_comparison_grouped(df_pubmed, df_preprints, title, filename, n, sort_by='pubmed'):
    """
    Create horizontal grouped bar chart comparing PubMed vs Preprints

    Shows actual TF-IDF scores for all terms, even if they're not in top-N for one source.
    This gives a more accurate comparison than setting missing terms to 0.

    Args:
        df_pubmed: DataFrame with PubMed TF-IDF results
        df_preprints: DataFrame with Preprints TF-IDF results
        title: Plot title
        filename: Output filename
        n: Number of top terms from each source to include
        sort_by: Sort order - 'pubmed' (default), 'preprints', or 'max' (by maximum score)

    Note: The plot shows the UNION of top-N from both sources, so if there's
    no overlap, up to 2*N terms may be displayed.
    """
    # Get top terms from both sources
    top_pm = df_pubmed.head(n)
    top_pp = df_preprints.head(n)

    # Combine and get unique terms (union of top-N from both sources)
    all_terms = set(top_pm['phrase'].tolist() + top_pp['phrase'].tolist())

    # Create lookup dictionaries from FULL dataframes (not just top-N)
    # This way we get actual TF-IDF scores even for low-ranking terms
    pm_dict = dict(zip(df_pubmed['phrase'], df_pubmed['mean_tfidf']))
    pp_dict = dict(zip(df_preprints['phrase'], df_preprints['mean_tfidf']))

    # Build comparison dataframe
    comparison_data = []
    for term in all_terms:
        # Get actual TF-IDF score (will be low if not in top-N, but not 0)
        pm_val = pm_dict.get(term, 0)
        pp_val = pp_dict.get(term, 0)
        max_val = max(pm_val, pp_val)
        comparison_data.append({
            'term': term,
            'PubMed': pm_val,
            'Preprints': pp_val,
            'max_score': max_val
        })

    comp_df = pd.DataFrame(comparison_data)

    # Sort based on specified source
    if sort_by == 'pubmed':
        comp_df = comp_df.sort_values('PubMed', ascending=True)  # ascending for horizontal bars
        sort_label = 'sorted by PubMed'
    elif sort_by == 'preprints':
        comp_df = comp_df.sort_values('Preprints', ascending=True)
        sort_label = 'sorted by Preprints'
    else:  # 'max'
        comp_df = comp_df.sort_values('max_score', ascending=True)
        sort_label = 'sorted by max score'

    num_terms = len(comp_df)

    # Adjust figure height based on number of terms for horizontal grouped bars
    # Each term needs ~0.30 inches of height (reduced from 0.35 to compress slightly)
    fig_height = max(8, num_terms * 0.30)

    # Create horizontal grouped bar plot
    fig, ax = plt.subplots(figsize=(11, fig_height))

    terms = comp_df['term'].tolist()
    y_pos = np.arange(len(terms))
    bar_height = 0.30  # Reduced from 0.35 to compress bars

    # Horizontal grouped bars with thick black borders
    ax.barh(y_pos - bar_height / 2, comp_df['PubMed'], bar_height,
            label='PubMed', color='#E63946', alpha=0.8,
            edgecolor='black', linewidth=1)
    ax.barh(y_pos + bar_height / 2, comp_df['Preprints'], bar_height,
            label='Preprints', color='#457B9D', alpha=0.8,
            edgecolor='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=12)  # Increased from 10 to 12
    ax.set_xlabel('Mean TF-IDF Score', fontsize=14, fontweight='bold')  # Increased from 13 to 14
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)  # Increased from 14 to 15
    ax.legend(loc='lower right', fontsize=13)  # Increased from 12 to 13
    ax.tick_params(axis='x', labelsize=11)  # Set x-axis tick label size
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    print(f"  Plotted {num_terms} unique terms (union of top-{n} from both sources, {sort_label})")


# ============================================================================
# DATA LOADING
# ============================================================================

print("\n[1/9] Loading data...")

# Load full dataset
df_cleaned = pd.read_csv('data/cleaned.csv', low_memory=False)
df_cleaned['text'] = df_cleaned.apply(get_text, axis=1)
df_cleaned = df_cleaned[df_cleaned['text'].str.len() > 1].copy()
print(f"  Loaded cleaned.csv: {len(df_cleaned)} articles")

# Load curated dataset
df_st2 = pd.read_csv('data/ST2.csv', low_memory=False)
filter_col = 'what section used' if 'what section used' in df_st2.columns else 'code'
df_st2 = df_st2[df_st2[filter_col].notna() & (df_st2[filter_col] != '')].copy()
df_st2['text'] = df_st2.apply(get_text, axis=1)
df_st2 = df_st2[df_st2['text'].str.len() > 10].copy()
df_st2['source_cat'] = df_st2['source'].apply(categorize_source)

print(f"  Loaded ST2.csv: {len(df_st2)} curated articles")
print(f"    PubMed: {len(df_st2[df_st2['source_cat'] == 'PubMed'])}")
print(f"    Preprints: {len(df_st2[df_st2['source_cat'] == 'Preprints'])}")

# ============================================================================
# STEP 1: FULL DATASET ANALYSIS (cleaned.csv)
# ============================================================================

print("\n[2/9] Analyzing FULL dataset (cleaned.csv)...")

full_tfidf = run_tfidf(df_cleaned, 'Full Dataset', max_features=1000)
full_tfidf.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/full_dataset_top{TOP_N}.csv', index=False)

print(f"  Top 5: {list(full_tfidf.head(5)['phrase'])}")

# ============================================================================
# STEP 2: CURATED DATASET - STANDARD TF-IDF
# ============================================================================

print("\n[3/9] Running STANDARD TF-IDF on curated dataset (ST2.csv)...")

standard_all = run_tfidf(df_st2, 'Curated ALL')
standard_pubmed = run_tfidf(df_st2[df_st2['source_cat'] == 'PubMed'], 'Curated PubMed')
standard_preprints = run_tfidf(df_st2[df_st2['source_cat'] == 'Preprints'], 'Curated Preprints')

standard_all.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/standard_ALL_top{TOP_N}.csv', index=False)
standard_pubmed.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/standard_PubMed_top{TOP_N}.csv', index=False)
standard_preprints.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/standard_Preprints_top{TOP_N}.csv', index=False)

# ============================================================================
# STEP 3: FILTERED TF-IDF (Remove generic terms)
# ============================================================================

print("\n[4/9] Running FILTERED TF-IDF (removing generic terms)...")

EXCLUDE_PATTERNS = [
    'large language', 'language model', 'llm', 'llms', 'generative ai',
    'deep learning', 'machine learning', 'artificial intelligence',
    'natural language', 'language processing', 'nlp',
    'state art', 'based', 'using',
    'https', 'github', 'model', 'learning', 'data'
]
# Note: Medical and genomic terms are NOT excluded per user request
# They are domain-specific and informative for this analysis

filtered_all = standard_all[~standard_all['phrase'].apply(lambda x: should_exclude(x, EXCLUDE_PATTERNS))].copy()
filtered_pubmed = standard_pubmed[
    ~standard_pubmed['phrase'].apply(lambda x: should_exclude(x, EXCLUDE_PATTERNS))].copy()
filtered_preprints = standard_preprints[
    ~standard_preprints['phrase'].apply(lambda x: should_exclude(x, EXCLUDE_PATTERNS))].copy()

print(f"  Filtered: {len(filtered_all)} phrases (removed {len(standard_all) - len(filtered_all)})")
print(f"  Top 5: {list(filtered_all.head(5)['phrase'])}")

filtered_all.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/filtered_ALL_top{TOP_N}.csv', index=False)
filtered_pubmed.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/filtered_PubMed_top{TOP_N}.csv', index=False)
filtered_preprints.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/filtered_Preprints_top{TOP_N}.csv', index=False)

# ============================================================================
# STEP 4: SOURCE COMPARISON
# ============================================================================

print("\n[5/9] Computing source overlap...")

std_pm_top = set(standard_pubmed.head(TOP_N)['phrase'])
std_pp_top = set(standard_preprints.head(TOP_N)['phrase'])
std_overlap = std_pm_top & std_pp_top

filt_pm_top = set(filtered_pubmed.head(TOP_N)['phrase'])
filt_pp_top = set(filtered_preprints.head(TOP_N)['phrase'])
filt_overlap = filt_pm_top & filt_pp_top

comparison_df = pd.DataFrame({
    'Analysis': ['Standard', 'Filtered'],
    'Top_N': [TOP_N, TOP_N],
    'PubMed_unique': [len(std_pm_top - std_pp_top), len(filt_pm_top - filt_pp_top)],
    'Preprints_unique': [len(std_pp_top - std_pm_top), len(filt_pp_top - filt_pm_top)],
    'Overlap': [len(std_overlap), len(filt_overlap)],
    'Overlap_pct': [len(std_overlap) / TOP_N * 100, len(filt_overlap) / TOP_N * 100],
    'Union_terms': [len(std_pm_top | std_pp_top), len(filt_pm_top | filt_pp_top)]
})

comparison_df.to_csv(f'{OUTPUT_DIR}/tables/source_comparison_top{TOP_N}.csv', index=False)
print(f"  Standard overlap: {len(std_overlap)}/{TOP_N} ({len(std_overlap) / TOP_N * 100:.0f}%)")
print(f"  Filtered overlap: {len(filt_overlap)}/{TOP_N} ({len(filt_overlap) / TOP_N * 100:.0f}%)")
print(f"  Standard union: {len(std_pm_top | std_pp_top)} unique terms")
print(f"  Filtered union: {len(filt_pm_top | filt_pp_top)} unique terms")

# ============================================================================
# STEP 5: COMBINED STACKED FIGURE (Fig0 + SFig1 + Fig1)
# ============================================================================

print("\n[6/9] Creating combined 3-panel figure (SFig1)...")

plot_three_horizontal_stacked(
    full_tfidf,
    standard_all,
    filtered_all,
    titles=[
        f'Full Dataset (Top {TOP_N})',
        f'Selected Articles (Top {TOP_N})',
        f'Selected Articles + Filtered Phrases (Top {TOP_N})'
    ],
    colors=['#1F77B4', '#6B8E23', '#2E86AB'],
    labels=['A', 'B', 'C'],
    filename=f'{OUTPUT_DIR}/figures_supplement/SFig1_progression_top{TOP_N}.pdf',
    n=TOP_N
)

# ============================================================================
# STEP 6: VISUALIZATIONS - SUPPLEMENT (Standard)
# ============================================================================

print("\n[7/9] Creating supplement source comparison (SFig2: Selected, before filtering)...")

plot_source_comparison_grouped(
    standard_pubmed,
    standard_preprints,
    f'Source Comparison: Selected Articles (Top-{TOP_N})\nBefore Filtering',
    f'{OUTPUT_DIR}/figures_supplement/SFig2_selected_comparison_top{TOP_N}.pdf',
    n=TOP_N,
    sort_by='pubmed'  # Sort by PubMed for consistency with Fig3
)

# ============================================================================
# STEP 6: VISUALIZATIONS - MAIN TEXT (Filtered)
# ============================================================================

print("\n[8/9] Creating main source comparison (Fig3: Selected + Filtered)...")

plot_source_comparison_grouped(
    filtered_pubmed,
    filtered_preprints,
    f'Source Comparison: Selected Articles + Filtered Phrases (Top-{TOP_N})\nSpecific Research Trends',
    f'{OUTPUT_DIR}/figures_main/Fig3_selected_filtered_comparison_top{TOP_N}.pdf',
    n=TOP_N,
    sort_by='pubmed'  # Sort by PubMed to highlight PubMed-specific clinical trends
)

# ============================================================================
# STEP 7: GENERATE SUMMARY
# ============================================================================

print("\n[9/9] Generating summary...")

summary = f"""# TF-IDF Analysis Results

## Full Dataset (cleaned.csv)
- Total articles: {len(df_cleaned)}
- Top 5 terms: {', '.join(full_tfidf.head(5)['phrase'].tolist())}

## Curated Dataset (ST2.csv)
- Total articles: {len(df_st2)}
- PubMed: {len(df_st2[df_st2['source_cat'] == 'PubMed'])}
- Preprints: {len(df_st2[df_st2['source_cat'] == 'Preprints'])}

## Standard Analysis (before filtering)
Top 10 phrases:
"""

for idx, (i, row) in enumerate(standard_all.head(10).iterrows(), 1):
    summary += f"{idx}. '{row['phrase']}' (TF-IDF: {row['mean_tfidf']:.4f}, n={row['doc_freq']} docs)\n"

summary += f"\nSource overlap: {len(std_overlap)}/{TOP_N} ({len(std_overlap) / TOP_N * 100:.0f}%)\n"
summary += f"Union of terms plotted in comparison: {len(std_pm_top | std_pp_top)}\n"

summary += f"""
## Filtered Analysis (generic terms removed)
Removed {len(standard_all) - len(filtered_all)} generic phrases

Top 10 specific trends:
"""

for idx, (i, row) in enumerate(filtered_all.head(10).iterrows(), 1):
    summary += f"{idx}. '{row['phrase']}' (TF-IDF: {row['mean_tfidf']:.4f}, n={row['doc_freq']} docs)\n"

summary += f"\nSource overlap: {len(filt_overlap)}/{TOP_N} ({len(filt_overlap) / TOP_N * 100:.0f}%)\n"
summary += f"Union of terms plotted in comparison: {len(filt_pm_top | filt_pp_top)}\n"

summary += """
## Interpretation

### Full Dataset Analysis
Shows the overall landscape of LLM-genomics research with 51,613 articles.
- Purpose: Demonstrate field coverage before curation
- Use: Context for understanding what literature exists

### Standard Analysis (Supplement)
Shows expected core topics: LLMs, deep learning, genomics, precision medicine.
- Purpose: Validate systematic coverage of our curated dataset
- Use: Supplementary figures to show we're comprehensive

### Filtered Analysis (Main Text)
Reveals specific trends beyond obvious keywords.
- Purpose: Show research diversity and maturity
- Use: Main text figures to demonstrate interesting insights

### Source Comparison
Grouped bar plots show direct comparison on same scale (top-{TOP_N} phrases).
- Overlap calculated from top-{TOP_N} phrases from each source
- Comparison plots show UNION of top-{TOP_N}, which can exceed {TOP_N} terms
- Standard analysis: Shows consensus on core topics
- Filtered analysis: Shows complementary contributions
- Conclusion: Both PubMed and Preprints provide unique value

### Hyperparameters
TF-IDF Configuration:
- ngram_range=(2, 3): Captures bigrams and trigrams for multi-word phrases
- max_features=1000: Balances comprehensiveness with computational efficiency
- stop_words: English stop words + custom ones (et al, etc.)
- Top-{TOP_N} phrases: Standardized across all analyses for consistency

### Source Comparison Notes
- Shows actual TF-IDF scores for all terms in the union of top-{TOP_N} from each source
- Union can contain up to {TOP_N*2} terms if no overlap exists
- Terms not in top-{TOP_N} of one source still show their actual (low) TF-IDF score, not 0
- This provides more accurate comparison of relative importance across sources
"""

with open(f'{OUTPUT_DIR}/ANALYSIS_SUMMARY.md', 'w') as f:
    f.write(summary)

print(f"  Saved: {OUTPUT_DIR}/ANALYSIS_SUMMARY.md")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nConfiguration: TOP_N = {TOP_N}")
print(f"\nOutputs:")
print(f"  Main figures (for paper): {OUTPUT_DIR}/figures_main/")
print(f"    - Fig3_selected_filtered_comparison_top{TOP_N}.pdf")
print(f"      PubMed vs Preprints comparison after selection + filtering")
print(f"")
print(f"  Supplement figures: {OUTPUT_DIR}/figures_supplement/")
print(f"    - SFig1_progression_top{TOP_N}.pdf")
print(f"      3-panel: (A) Full dataset, (B) Selected Articles, (C) Selected Articles + Filtered Phrases")
print(f"    - SFig2_selected_comparison_top{TOP_N}.pdf")
print(f"      PubMed vs Preprints comparison after selection (before filtering)")
print(f"")
print(f"  Data tables: {OUTPUT_DIR}/tables/")
print(f"    - 7 CSV files (full + standard + filtered, all top-{TOP_N})")
print(f"    - source_comparison_top{TOP_N}.csv (overlap statistics)")
print(f"")
print(f"  Summary: {OUTPUT_DIR}/ANALYSIS_SUMMARY.md")
print("\n" + "=" * 80)
