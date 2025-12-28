# Figure Captions for TF-IDF Analysis

**NOTE:** All TF-IDF figures are now in SUPPLEMENT ONLY (no main text figures per reviewer request).

---

## Supplementary Material

### Supplementary Figure S1. Progression of TF-IDF Analysis from Full Dataset to Filtered Insights

Three-panel progression showing the systematic refinement of research topics through article selection and term filtering. **(A) Full Dataset (51,613 articles):** Broad landscape dominated by generic terms including "language models" (TF-IDF=0.0200), "large language" (0.0200), and "artificial intelligence" (0.0174), validating comprehensive field coverage. **(B) Selected Articles (195 articles):** Curated dataset after manual review shows similar term dominance ("large language", TF-IDF=0.0455; "language models", 0.0455) but with emerging domain-specific concepts like "precision medicine" and "gene expression", confirming systematic selection captured core literature. **(C) Selected + Filtered:** Removal of 193 generic AI/ML terms reveals specific research foci: 'precision medicine' (TF-IDF=0.0172), 'gene expression' (0.0151), 'genetic testing' (0.0124), disease-specific applications ('breast cancer'), knowledge resources ('human phenotype ontology'), and emerging techniques ('attention mechanism', 'single cell'). All panels display top-30 phrases with horizontal bars for readability, with panel labels A, B, C positioned at top-left. This progression demonstrates the value of systematic filtering to uncover specific research trends beyond obvious keywords.

### Supplementary Figure S2. Source Comparison of Research Trends After Article Selection

Source-specific research trends comparing PubMed (n=131) and preprints (n=64) after article selection but before filtering generic terms. Vertical grouped bars show actual TF-IDF scores for top-30 phrases from each source on the same scale. The plot displays 43 unique terms (union of top-30 from both sources), showing 57% overlap (17/30 shared phrases). Strong consensus exists on core concepts: both sources emphasize "large language" and "language models", "artificial intelligence", and "generative ai". This high overlap validates that our curated dataset systematically captured central themes from both publication types. Comparison with Supplementary Figure S3 (after filtering) reveals that generic term removal reduces overlap from 57% to 23%, exposing complementary source-specific contributions. Terms show actual TF-IDF scores from the full analysis, providing accurate relative importance even for phrases not in both top-30 lists.

---

### Supplementary Figure S3. Source Comparison After Selection and Filtering

Source-specific research trends comparing PubMed (n=131) and preprints (n=64) after article selection and filtering generic AI/ML terms. Vertical grouped bars show actual TF-IDF scores for the union of top-30 phrases from both sources. The plot displays 53 unique terms, demonstrating 23% overlap (7/30 shared phrases). PubMed emphasizes clinical applications ('precision medicine', 'genetic testing') and established medical concepts, while preprints focus on emerging computational methods ('gene expression', 'open source') and novel techniques. This complementary coverage demonstrates that both publication types contribute unique perspectives to the field.

### Supplementary Figure S4. Topic Modeling Scatter

Topic modeling visualization showing overlap and relationships between themes. Eight LDA topics fitted on the curated dataset (n=195) are displayed in 2D space using Jensen-Shannon divergence-based layout. Bubble size reflects topic prevalence. Labels show top five terms per topic. Topics closer together share more vocabulary, illustrating the semantic landscape of LLM applications in medical genomics.

### Supplementary Figure S5. Fine-Tuned Source Comparison

Source comparison using fine-tuned analysis (context preserved, post-hoc reweighting). This analysis addresses Reviewer Comment #2 by first training TF-IDF with full context, then down-weighting generic AI/ML terms. Results confirm that domain-specific trends remain stable (23% overlap between sources), validating the filtered analysis approach.

---

## Notes for Authors

- **All TF-IDF figures are now in SUPPLEMENT ONLY** (no main text figures per reviewer request)
- **Supplementary Figures S1-S5** should appear in supplement
- All figures use consistent top-30 ranking for comparability
- Comparison plots show UNION of top-30 terms (can display 30-60 terms depending on overlap)
- Color scheme: Blue (#1F77B4) for full dataset and filtered analysis, olive (#6B8E23) for selected articles, red (#E63946) for PubMed, blue (#457B9D) for preprints
- All text sized for readability: labels 11pt, titles 14pt, axis labels 13pt

## Files

- `results/figures_supplement/SFig1_progression_top30.pdf`
- `results/figures_supplement/SFig2_selected_comparison_top30.pdf`
- `results/figures_supplement/SFig3_selected_filtered_comparison_top30.pdf`
- `results/figures_supplement/SFig4_topic_scatter.pdf`
- `results/figures_supplement/SFig5_finetuned_comparison_top30.pdf`
