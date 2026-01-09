# TF-IDF Analysis Results

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
analysis on the curated dataset. Each row represents one of 8 topics identified through unsupervised topic
modeling, including: topic identifier, 2D coordinates for visualization (derived from Jensen-Shannon divergence),
topic prevalence (proportion of corpus assigned to this topic), and the top 5 most representative
terms for each topic. This table supports the topic modeling scatter plot (Supplementary Figure 4) and provides
insight into the semantic structure of the included literature.

**Supplementary Table 4 (ST4)** presents the document-topic probability distribution matrix from the LDA analysis.
Each row corresponds to one article from ST2, with columns representing the probability that the article belongs to
each of the 8 identified topics. An additional 'source' column categorizes articles as PubMed, Preprints,
or Other. This table enables analysis of topic distribution patterns across different publication venues and
supports reproducibility of the topic modeling results.

Detailed descriptions of column meanings and classification codes are available in the project GitHub repository
(https://github.com/TohaRhymes/llm_in_diagnostics).

## Full Dataset (cleaned.csv)
- Total articles: 51613
- Top 5 terms: language models, large language, artificial intelligence, deep learning, covid 19

## Curated Dataset (ST2.csv)
- Total articles: 325
- PubMed: 229
- Preprints: 96

## Standard Analysis (before filtering)
Top 10 phrases:
1. 'large language' (TF-IDF: 0.0395, n=94 docs)
2. 'language models' (TF-IDF: 0.0393, n=92 docs)
3. 'large language models' (TF-IDF: 0.0333, n=77 docs)
4. 'deep learning' (TF-IDF: 0.0289, n=67 docs)
5. 'artificial intelligence' (TF-IDF: 0.0280, n=67 docs)
6. 'natural language' (TF-IDF: 0.0263, n=76 docs)
7. 'natural language processing' (TF-IDF: 0.0247, n=72 docs)
8. 'language processing' (TF-IDF: 0.0247, n=72 docs)
9. 'machine learning' (TF-IDF: 0.0204, n=48 docs)
10. 'state art' (TF-IDF: 0.0193, n=53 docs)

Source overlap: 20/30 (67%)
Union of terms plotted in comparison: 40

## Filtered Analysis (generic terms removed)
Removed 220 generic phrases

Top 10 specific trends:
1. 'genetic testing' (TF-IDF: 0.0183, n=28 docs)
2. 'pre trained' (TF-IDF: 0.0175, n=33 docs)
3. 'gene expression' (TF-IDF: 0.0151, n=20 docs)
4. 'precision medicine' (TF-IDF: 0.0142, n=21 docs)
5. 'electronic health' (TF-IDF: 0.0134, n=30 docs)
6. 'intelligence ai' (TF-IDF: 0.0134, n=36 docs)
7. 'attention mechanism' (TF-IDF: 0.0126, n=26 docs)
8. 'genetic variants' (TF-IDF: 0.0114, n=21 docs)
9. 'health records' (TF-IDF: 0.0112, n=24 docs)
10. 'clinical trials' (TF-IDF: 0.0112, n=15 docs)

Source overlap: 10/30 (33%)
Union of terms plotted in comparison: 50

## Fine-Tuned Analysis (context preserved, post-hoc reweighting)
Top 10 phrases:
1. 'pre trained' (TF-IDF: 0.0214, n=33 docs)
2. 'genetic testing' (TF-IDF: 0.0202, n=28 docs)
3. 'gene expression' (TF-IDF: 0.0173, n=20 docs)
4. 'intelligence ai' (TF-IDF: 0.0162, n=36 docs)
5. 'precision medicine' (TF-IDF: 0.0162, n=21 docs)
6. 'electronic health' (TF-IDF: 0.0161, n=30 docs)
7. 'attention mechanism' (TF-IDF: 0.0145, n=26 docs)
8. 'fine tuned' (TF-IDF: 0.0137, n=16 docs)
9. 'health records' (TF-IDF: 0.0137, n=24 docs)
10. 'genetic variants' (TF-IDF: 0.0132, n=21 docs)

Source overlap: 10/30 (33%)
Union of terms plotted in comparison: 50

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

### Topic Modeling Scatter (Supplementary Figure S5)
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
- Added Supplementary Figure S5 (topic scatter) summarizing eight LDA topics with PCA coordinates and prevalence-based bubble sizes.
- Scatter labels display the top five bigram terms per topic, making overlap intuitive without adding another set of bar charts.
- We can now state in the response that topic modeling visual support is provided while keeping the figure count lean (one new supplementary figure covering both requests).
