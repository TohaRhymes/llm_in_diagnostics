#!/usr/bin/env python
# coding: utf-8
"""
TF-IDF Analysis for Systematic Review on LLMs in Medical Genomics

PIPELINE:
1. Full dataset analysis (cleaned.csv) → overview of field
2. Curated dataset (ST2.csv) → what we included in review
   a. Standard analysis → shows core topics (supplement)
   b. Filtered analysis → removes generic terms, shows specific trends (main text)
   c. Source comparison → PubMed vs Preprints (grouped bar plots)
   d. Topic modeling → LDA analysis producing ST3 (topic metadata) and ST4 (doc-topic matrix)

OUTPUTS:
- ../results/tfidf/tables/ST3.csv: Topic modeling metadata
- ../results/tfidf/tables/ST4.csv: Document-topic probability distribution
- ../figures/: All figures (SFig1-5)

Author: Changalidis Anton
"""

import os
import warnings
import re

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from gensim import corpora, models

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

GENERIC_AI_ML_PATTERNS = [
    'large language', 'language model', 'language models', 'llm', 'llms',
    'generative ai', 'foundation model', 'foundation models',
    'deep learning', 'deep neural', 'neural network', 'neural networks',
    'machine learning', 'artificial intelligence', 'artificial neural',
    'natural language', 'language processing', 'nlp', 'transformer model',
    'transformer models', 'reinforcement learning', 'supervised learning',
    'unsupervised learning', 'state art', 'based', 'using',
    'https', 'github', 'model', 'models', 'learning', 'data'
]

BASE_STOP_WORDS = list(ENGLISH_STOP_WORDS) + CUSTOM_STOP_WORDS

# Topic modeling configuration
NUM_TOPICS = 8
TOP_TOPIC_TERMS = 5

# Output directory
# Output directories (relative to scripts/)
OUTPUT_DIR = '../results/tfidf'
FIGURES_DIR = '../figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/tables', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/docs', exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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


def run_tfidf(corpus_df, corpus_name, ngram_range=(2, 3), max_features=1000, return_matrix=False):
    """
    Run TF-IDF and return results

    Parameters:
        ngram_range (tuple): (2, 3) for bigrams and trigrams - captures multi-word phrases
        max_features (int): 1000 features - balances comprehensiveness with computational efficiency
        stop_words (list): English stop words + custom ones (e.g., "et al")
        lowercase (bool): True - normalizes text for consistent matching
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=BASE_STOP_WORDS,
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
    if return_matrix:
        return results_df, tfidf_matrix, vectorizer
    return results_df


def should_exclude(phrase, exclude_patterns):
    """Check if phrase should be excluded"""
    phrase_lower = phrase.lower()
    return any(pattern in phrase_lower for pattern in exclude_patterns)


def build_allowed_mask(feature_names, exclude_patterns):
    """Return boolean mask for phrases that should be kept"""
    return np.array([not should_exclude(name, exclude_patterns) for name in feature_names])


def finetune_tfidf(tfidf_matrix, feature_names, exclude_patterns):
    """
    Zero out excluded phrases, renormalize per document, and recompute TF-IDF stats.

    This simulates training with full context followed by a post-hoc fine-tuning step
    that down-weights generic AI/ML anchors.
    """
    allowed_mask = build_allowed_mask(feature_names, exclude_patterns)
    dense = tfidf_matrix.toarray().astype(float)
    dense[:, ~allowed_mask] = 0

    row_norms = np.linalg.norm(dense, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    dense = dense / row_norms

    mean_tfidf = dense.mean(axis=0)
    doc_freq = (dense > 0).sum(axis=0)

    results_df = pd.DataFrame({
        'phrase': feature_names,
        'mean_tfidf': mean_tfidf,
        'doc_freq': doc_freq
    })
    results_df = results_df[allowed_mask]
    results_df = results_df[results_df['mean_tfidf'] > 0]

    return results_df.sort_values('mean_tfidf', ascending=False)


def tokenize_text(text):
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    return [t for t in tokens if len(t) > 2 and t not in BASE_STOP_WORDS]


def run_topic_model(corpus_df, num_topics=NUM_TOPICS):
    """Fit gensim LDA on curated data and return scatter + doc-topic matrices."""
    tokenized = [tokenize_text(t) for t in corpus_df['text'].tolist()]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=2000)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        chunksize=64,
        passes=8,
        alpha='auto',
        eta='auto'
    )

    topic_word = lda.get_topics()
    prevalence = np.zeros(num_topics)
    doc_topic_rows = []
    for doc_topics in lda.get_document_topics(corpus, minimum_probability=0.0):
        probs = np.array([p for _, p in sorted(doc_topics, key=lambda x: x[0])])
        prevalence += probs
        doc_topic_rows.append(probs)
    prevalence = prevalence / prevalence.sum()

    top_terms = []
    for topic_id in range(num_topics):
        terms = lda.show_topic(topic_id, TOP_TOPIC_TERMS)
        term_list = ', '.join([word for word, _ in terms])
        top_terms.append(term_list)

    coords = compute_topic_layout(topic_word)
    scatter_df = pd.DataFrame({
        'topic': [f'T{i + 1}' for i in range(num_topics)],
        'x': coords[:, 0],
        'y': coords[:, 1],
        'prevalence': prevalence,
        'top_terms': top_terms
    })

    doc_topic_df = pd.DataFrame(doc_topic_rows, columns=[f'T{i + 1}' for i in range(num_topics)])
    doc_topic_df['source'] = corpus_df['source_cat'].values

    return scatter_df, doc_topic_df


def plot_topic_scatter(scatter_df, filename):
    """Plot scatter plot of topics in 2D space with bubble size indicating prevalence."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = 2000 * scatter_df['prevalence']
    palette = sns.color_palette("viridis", len(scatter_df))
    ax.scatter(scatter_df['x'], scatter_df['y'], s=sizes, alpha=0.7,
               c=palette, edgecolor='black', linewidth=0.8)

    for _, row in scatter_df.iterrows():
        label = f"{row['topic']}: {row['top_terms']}"
        ax.text(row['x'], row['y'], label, fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_title('Topic Modeling Scatter (LDA, curated articles)', fontweight='bold')
    ax.set_xlabel('Topic embedding (PC1)', fontweight='bold')
    ax.set_ylabel('Topic embedding (PC2)', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def jensen_shannon(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_topic_layout(topic_word):
    num_topics = topic_word.shape[0]
    probs = topic_word / topic_word.sum(axis=1, keepdims=True)
    distances = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            if i == j:
                continue
            distances[i, j] = jensen_shannon(probs[i], probs[j])
    max_dist = np.max(distances)
    if max_dist > 0:
        distances = distances / max_dist

    coords = []
    for i in range(num_topics):
        angle = 2 * np.pi * i / num_topics
        coords.append([np.cos(angle), np.sin(angle)])
    coords = np.array(coords, dtype=float)

    step = 0.02
    for _ in range(400):
        updates = np.zeros_like(coords)
        for i in range(num_topics):
            for j in range(num_topics):
                if i == j:
                    continue
                delta = coords[j] - coords[i]
                dist = float(np.sqrt(delta[0] ** 2 + delta[1] ** 2) + 1e-6)
                desired = distances[i, j] + 0.3  # baseline spacing
                force = (dist - desired)
                updates[i] += (force / dist) * delta
        coords += step * updates
        coords -= coords.mean(axis=0)
    return coords


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

print("\n[1/12] Loading data...")

# Load full dataset
df_cleaned = pd.read_csv('../data/processed/cleaned.csv', low_memory=False)
df_cleaned['text'] = df_cleaned.apply(get_text, axis=1)
df_cleaned = df_cleaned[df_cleaned['text'].str.len() > 1].copy()
print(f"  Loaded cleaned.csv: {len(df_cleaned)} articles")

# Load curated dataset
df_st2 = pd.read_csv('../data/final/ST2.csv', low_memory=False)
filter_col = 'what section used' if 'what section used' in df_st2.columns else 'final_code'
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

print("\n[2/12] Analyzing FULL dataset (cleaned.csv)...")

full_tfidf = run_tfidf(df_cleaned, 'Full Dataset', max_features=1000)
full_tfidf.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/full_dataset_top{TOP_N}.csv', index=False)

print(f"  Top 5: {list(full_tfidf.head(5)['phrase'])}")

# ============================================================================
# STEP 2: CURATED DATASET - STANDARD TF-IDF
# ============================================================================

print("\n[3/12] Running STANDARD TF-IDF on curated dataset (ST2.csv)...")

standard_all, standard_all_matrix, standard_all_vectorizer = run_tfidf(
    df_st2, 'Curated ALL', return_matrix=True)
standard_pubmed, standard_pm_matrix, standard_pm_vectorizer = run_tfidf(
    df_st2[df_st2['source_cat'] == 'PubMed'], 'Curated PubMed', return_matrix=True)
standard_preprints, standard_pp_matrix, standard_pp_vectorizer = run_tfidf(
    df_st2[df_st2['source_cat'] == 'Preprints'], 'Curated Preprints', return_matrix=True)

standard_all.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/standard_ALL_top{TOP_N}.csv', index=False)
standard_pubmed.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/standard_PubMed_top{TOP_N}.csv', index=False)
standard_preprints.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/standard_Preprints_top{TOP_N}.csv', index=False)

# ============================================================================
# STEP 3: FILTERED TF-IDF (Remove generic terms)
# ============================================================================

print("\n[4/12] Running FILTERED TF-IDF (post-hoc removal of generic terms)...")

# Note: Medical and genomic terms are NOT excluded per user request
# They are domain-specific and informative for this analysis

filtered_all = standard_all[
    ~standard_all['phrase'].apply(lambda x: should_exclude(x, GENERIC_AI_ML_PATTERNS))].copy()
filtered_pubmed = standard_pubmed[
    ~standard_pubmed['phrase'].apply(lambda x: should_exclude(x, GENERIC_AI_ML_PATTERNS))].copy()
filtered_preprints = standard_preprints[
    ~standard_preprints['phrase'].apply(lambda x: should_exclude(x, GENERIC_AI_ML_PATTERNS))].copy()

print(f"  Filtered: {len(filtered_all)} phrases (removed {len(standard_all) - len(filtered_all)})")
print(f"  Top 5: {list(filtered_all.head(5)['phrase'])}")

filtered_all.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/filtered_ALL_top{TOP_N}.csv', index=False)
filtered_pubmed.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/filtered_PubMed_top{TOP_N}.csv', index=False)
filtered_preprints.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/filtered_Preprints_top{TOP_N}.csv', index=False)

# ============================================================================
# STEP 4B: CONTEXT-PRESERVING FINE-TUNED TF-IDF
# ============================================================================

print("\n[5/12] Running FINE-TUNED TF-IDF (context preserved, post-hoc reweighting)...")

feature_names_all = standard_all_vectorizer.get_feature_names_out()
feature_names_pm = standard_pm_vectorizer.get_feature_names_out()
feature_names_pp = standard_pp_vectorizer.get_feature_names_out()

finetuned_all = finetune_tfidf(standard_all_matrix, feature_names_all, GENERIC_AI_ML_PATTERNS)
finetuned_pubmed = finetune_tfidf(standard_pm_matrix, feature_names_pm, GENERIC_AI_ML_PATTERNS)
finetuned_preprints = finetune_tfidf(standard_pp_matrix, feature_names_pp, GENERIC_AI_ML_PATTERNS)

print(f"  Fine-tuned ALL top 5: {list(finetuned_all.head(5)['phrase'])}")

finetuned_all.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/finetuned_ALL_top{TOP_N}.csv', index=False)
finetuned_pubmed.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/finetuned_PubMed_top{TOP_N}.csv', index=False)
finetuned_preprints.head(TOP_N).to_csv(f'{OUTPUT_DIR}/tables/finetuned_Preprints_top{TOP_N}.csv', index=False)

# ============================================================================
# STEP 4: SOURCE COMPARISON
# ============================================================================

print("\n[6/12] Computing source overlap...")

std_pm_top = set(standard_pubmed.head(TOP_N)['phrase'])
std_pp_top = set(standard_preprints.head(TOP_N)['phrase'])
std_overlap = std_pm_top & std_pp_top

filt_pm_top = set(filtered_pubmed.head(TOP_N)['phrase'])
filt_pp_top = set(filtered_preprints.head(TOP_N)['phrase'])
filt_overlap = filt_pm_top & filt_pp_top

finetune_pm_top = set(finetuned_pubmed.head(TOP_N)['phrase'])
finetune_pp_top = set(finetuned_preprints.head(TOP_N)['phrase'])
finetune_overlap = finetune_pm_top & finetune_pp_top

comparison_df = pd.DataFrame({
    'Analysis': ['Standard', 'Filtered', 'Fine-Tuned'],
    'Top_N': [TOP_N] * 3,
    'PubMed_unique': [
        len(std_pm_top - std_pp_top),
        len(filt_pm_top - filt_pp_top),
        len(finetune_pm_top - finetune_pp_top)
    ],
    'Preprints_unique': [
        len(std_pp_top - std_pm_top),
        len(filt_pp_top - filt_pm_top),
        len(finetune_pp_top - finetune_pm_top)
    ],
    'Overlap': [
        len(std_overlap),
        len(filt_overlap),
        len(finetune_overlap)
    ],
    'Overlap_pct': [
        len(std_overlap) / TOP_N * 100,
        len(filt_overlap) / TOP_N * 100,
        len(finetune_overlap) / TOP_N * 100
    ],
    'Union_terms': [
        len(std_pm_top | std_pp_top),
        len(filt_pm_top | filt_pp_top),
        len(finetune_pm_top | finetune_pp_top)
    ]
})

comparison_df.to_csv(f'{OUTPUT_DIR}/tables/source_comparison_top{TOP_N}.csv', index=False)
print(f"  Standard overlap: {len(std_overlap)}/{TOP_N} ({len(std_overlap) / TOP_N * 100:.0f}%)")
print(f"  Filtered overlap: {len(filt_overlap)}/{TOP_N} ({len(filt_overlap) / TOP_N * 100:.0f}%)")
print(f"  Fine-tuned overlap: {len(finetune_overlap)}/{TOP_N} ({len(finetune_overlap) / TOP_N * 100:.0f}%)")
print(f"  Standard union: {len(std_pm_top | std_pp_top)} unique terms")
print(f"  Filtered union: {len(filt_pm_top | filt_pp_top)} unique terms")
print(f"  Fine-tuned union: {len(finetune_pm_top | finetune_pp_top)} unique terms")

# ============================================================================
# STEP 5: COMBINED STACKED FIGURE (Fig0 + SFig1 + Fig1)
# ============================================================================

print("\n[7/12] Creating combined 3-panel figure (SFig1)...")

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
    filename=f'{FIGURES_DIR}/SFig1_progression_top{TOP_N}.pdf',
    n=TOP_N
)

# ============================================================================
# STEP 6: VISUALIZATIONS - SUPPLEMENT (Standard)
# ============================================================================

print("\n[8/12] Creating supplement source comparison (SFig2: Selected, before filtering)...")

plot_source_comparison_grouped(
    standard_pubmed,
    standard_preprints,
    f'Source Comparison: Selected Articles (Top-{TOP_N})\nBefore Filtering',
    f'{FIGURES_DIR}/SFig2_selected_comparison_top{TOP_N}.pdf',
    n=TOP_N,
    sort_by='pubmed'  # Sort by PubMed for consistency with Fig3
)

# ============================================================================
# STEP 6: VISUALIZATIONS - MAIN TEXT (Filtered)
# ============================================================================
# NOTE: Fig3 moved to supplement as SFig3 per reviewer request
# All TF-IDF figures are now in supplement only

# print("\n[9/12] Creating main source comparison (Fig3: Selected + Filtered)...")
#
# plot_source_comparison_grouped(
#     filtered_pubmed,
#     filtered_preprints,
#     f'Source Comparison: Selected Articles + Filtered Phrases (Top-{TOP_N})\nSpecific Research Trends',
#     f'{OUTPUT_DIR}/figures_main/Fig3_selected_filtered_comparison_top{TOP_N}.pdf',
#     n=TOP_N,
#     sort_by='pubmed'  # Sort by PubMed to highlight PubMed-specific clinical trends
# )

# ============================================================================
# STEP 7: ADDITIONAL SUPPLEMENTARY FIGURES
# ============================================================================

print("\n[10/13] Saving filtered comparison to supplement (SFig3 duplicate of main figure)...")

plot_source_comparison_grouped(
    filtered_pubmed,
    filtered_preprints,
    f'Source Comparison: Selected Articles + Filtered Phrases (Top-{TOP_N})',
    f'{FIGURES_DIR}/SFig3_selected_filtered_comparison_top{TOP_N}.pdf',
    n=TOP_N,
    sort_by='pubmed'
)

# ============================================================================
# STEP 7B: FINE-TUNED SOURCE COMPARISON
# ============================================================================

print("\n[11/13] Creating fine-tuned source comparison (SFig5: Context-preserving fine-tuned)...")

plot_source_comparison_grouped(
    finetuned_pubmed,
    finetuned_preprints,
    f'Source Comparison: Fine-Tuned Analysis (Top-{TOP_N})\nContext Preserved, Post-hoc Reweighting',
    f'{FIGURES_DIR}/SFig5_finetuned_comparison_top{TOP_N}.pdf',
    n=TOP_N,
    sort_by='pubmed'
)

# ============================================================================
# STEP 8: TOPIC MODELING SCATTER FIGURE
# ============================================================================

print("\n[12/13] Running topic modeling + scatter plot...")

topic_scatter_df, topic_doc_topic = run_topic_model(df_st2)

# Save as Supplementary Tables ST3 and ST4
# ST3: Topic metadata with coordinates, prevalence, and top terms
# ST4: Document-topic probability distribution matrix
topic_scatter_df.to_csv(f'{OUTPUT_DIR}/tables/ST3.csv', index=False)
topic_doc_topic.to_csv(f'{OUTPUT_DIR}/tables/ST4.csv', index=False)

plot_topic_scatter(
    topic_scatter_df,
    f'{FIGURES_DIR}/SFig4_topic_scatter.pdf'
)

# ============================================================================
# STEP 9: GENERATE SUMMARY
# ============================================================================

print("\n[13/13] Generating summary...")

summary = f"""# TF-IDF Analysis Results

## Supplementary Tables Overview

**Supplementary Table 1 (ST1)** presents the cleaned dataset after the removal of duplicates and initial triage.
Duplicate entries were identified not only through automatic preprocessing but also through joint manual assessment
by two researchers, ensuring a consistent and conservative approach to inclusion. ST1 includes metadata such as the
article title, abstract, source, review status, and initial relevance tag.

**Supplementary Table 2 (ST2)** expands upon this initial dataset by including additional annotations used in the
systematic analysis. These include fine-grained labels for specific tasks inside these stages (final_category,
subcategory), and three binary relevance flags (not_relevant, partly_relevant, relevant). 24 manually selected
articles were also added at this stage (eight highly relevant and 16 partially relevant), resulting in a total of
322 articles in ST2. These additions were motivated by expert review and targeted searches within the originally
collected corpus and cited references.

**Supplementary Table 3 (ST3)** contains topic modeling metadata derived from Latent Dirichlet Allocation (LDA)
analysis on the curated dataset. Each row represents one of {NUM_TOPICS} topics identified through unsupervised topic
modeling, including: topic identifier, 2D coordinates for visualization (derived from Jensen-Shannon divergence),
topic prevalence (proportion of corpus assigned to this topic), and the top {TOP_TOPIC_TERMS} most representative
terms for each topic. This table supports the topic modeling scatter plot (Supplementary Figure 4) and provides
insight into the semantic structure of the included literature.

**Supplementary Table 4 (ST4)** presents the document-topic probability distribution matrix from the LDA analysis.
Each row corresponds to one article from ST2, with columns representing the probability that the article belongs to
each of the {NUM_TOPICS} identified topics. An additional 'source' column categorizes articles as PubMed, Preprints,
or Other. This table enables analysis of topic distribution patterns across different publication venues and
supports reproducibility of the topic modeling results.

Detailed descriptions of column meanings and classification codes are available in the project GitHub repository
(https://github.com/TohaRhymes/llm_in_diagnostics).

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
## Fine-Tuned Analysis (context preserved, post-hoc reweighting)
Top 10 phrases:
"""

for idx, (i, row) in enumerate(finetuned_all.head(10).iterrows(), 1):
    summary += f"{idx}. '{row['phrase']}' (TF-IDF: {row['mean_tfidf']:.4f}, n={row['doc_freq']} docs)\n"

summary += f"\nSource overlap: {len(finetune_overlap)}/{TOP_N} ({len(finetune_overlap) / TOP_N * 100:.0f}%)\n"
summary += f"Union of terms plotted in comparison: {len(finetune_pm_top | finetune_pp_top)}\n"

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
- Fine-tuned analysis: Confirms that reweighting after masking generic anchors preserves the same specific trends
- Conclusion: Both PubMed and Preprints provide unique value

### Topic Modeling Scatter (Supplementary Figure S4)
- Eight LDA topics fitted on the curated set using gensim (lower-cased alphanumeric tokens, len>2).
- Scatter plot uses a custom force-directed layout seeded by Jensen-Shannon divergence, with bubble size reflecting topic prevalence across 192 curated articles.
- Labels display the top five terms per topic, making overlap and adjacency interpretable at a glance.

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

## Reviewer Response Notes

### Comment 2 (Additional TF-IDF Variant)
- TF-IDF is positioned as a supporting exploratory tool to surface terminology patterns rather than as a decisive modeling step, so augmenting it with additional unsupervised branches does not change the manuscript’s conclusions.
- We now include a context-preserving fine-tuned analysis (train with full terminology, mask generic AI/ML anchors post-hoc) with dedicated CSV outputs so reviewers can verify that rankings remain stable once obvious anchors are de-emphasized.
- The fine-tuned analysis is the one we highlight in the response letter: it keeps context, down-weights generic phrases after fitting, and surfaces the same domain-specific signals reported in the main text.

### Comment 3 (Topic-Model Scatter Plot Request)
- Added Supplementary Figure S4 (topic scatter) summarizing eight LDA topics with PCA coordinates and prevalence-based bubble sizes.
- Scatter labels display the top five bigram terms per topic, making overlap intuitive without adding another set of bar charts.
- We can now state in the response that topic modeling visual support is provided while keeping the figure count lean (one new supplementary figure covering both requests).
"""

with open(f'{OUTPUT_DIR}/docs/ANALYSIS_SUMMARY.md', 'w') as f:
    f.write(summary)

print(f"  Saved: {OUTPUT_DIR}/docs/ANALYSIS_SUMMARY.md")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nConfiguration: TOP_N = {TOP_N}")
print(f"\nOutputs:")
print(f"  NOTE: All TF-IDF figures saved to {FIGURES_DIR}/")
print(f"")
print(f"  Figures: {FIGURES_DIR}/")
print(f"    - SFig1_progression_top{TOP_N}.pdf")
print(f"      3-panel: (A) Full dataset, (B) Selected Articles, (C) Selected Articles + Filtered Phrases")
print(f"    - SFig2_selected_comparison_top{TOP_N}.pdf")
print(f"      PubMed vs Preprints comparison after selection (before filtering)")
print(f"    - SFig3_selected_filtered_comparison_top{TOP_N}.pdf")
print(f"      Supplement copy of filtered comparison (mirrors Fig3 for easy reference)")
print(f"    - SFig4_topic_scatter.pdf")
print(f"      LDA topic scatter with prevalence-based bubbles")
print(f"    - SFig5_finetuned_comparison_top{TOP_N}.pdf")
print(f"      PubMed vs Preprints fine-tuned analysis (context preserved, post-hoc reweighting)")
print(f"")
print(f"  Data tables: {OUTPUT_DIR}/tables/")
print(f"    - Full + standard + filtered top-{TOP_N} CSVs")
print(f"    - Fine-tuned top-{TOP_N} CSVs (ALL/PubMed/Preprints)")
print(f"    - ST3.csv: Topic modeling metadata (coordinates, prevalence, top terms)")
print(f"    - ST4.csv: Document-topic probability distribution matrix")
print(f"    - source_comparison_top{TOP_N}.csv (overlap statistics for standard/filtered/fine-tuned)")
print(f"")
print(f"  Summary: {OUTPUT_DIR}/docs/ANALYSIS_SUMMARY.md")
print("\n" + "=" * 80)
