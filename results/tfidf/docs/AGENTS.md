# Repository Guidelines

## Project Structure & Module Organization
`tfidf_analysis.py` is the single entry point; it loads corpora from `data/cleaned.csv` (51,613 articles) and `data/ST2_v2.csv` (187 curated articles). Generated artifacts live under `results/` with `figures_main/`, `figures_supplement/`, and `tables/` plus Markdown summaries such as `ANALYSIS_SUMMARY.md` and `FIGURE_CAPTIONS.md`. Keep exploratory notebooks (e.g., `Untitled.ipynb`) isolated and commit only deterministic scripts or documentation.

## Build, Test, and Development Commands
- `python tfidf_analysis.py`: runs the full TF-IDF pipeline, overwriting figures and tables according to the current `TOP_N` constant.
- `python -m compileall tfidf_analysis.py`: quick syntax check before committing.
- `python tfidf_analysis.py && ls results/tables`: verify that every expected CSV (full, standard, filtered, comparison) is refreshed. Run inside a clean virtualenv with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn` installed.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` identifiers, and module-level constants in ALL_CAPS (e.g., `TOP_N`, `CUSTOM_STOP_WORDS`). Keep helper functions pure and documented via short docstrings explaining inputs/outputs. Prefer expressive filenames such as `SFig1_progression_top30.pdf`; when adding outputs, mirror this descriptive pattern and include the `TOP_N` value so downstream scripts remain traceable.

## Testing Guidelines
There is no separate test suite, so treat every pipeline run as an integration test. After executing `python tfidf_analysis.py`, confirm counts printed in the console (e.g., 51,613 full articles, 187 curated) and open a sample PDF/CSV to ensure plots and tables were regenerated. When modifying preprocessing rules or stop-word lists, spot-check TF-IDF rankings for regressions and record notable shifts in `ANALYSIS_SUMMARY.md`.

## Commit & Pull Request Guidelines
Recent history favors concise, present-tense subjects (`tfidf + hist usage`, `imgs updated`). Follow that style, grouping related file changes per commit. Pull requests should summarize the motivation, list regenerated artifacts (figures/tables), reference data snapshots or issue IDs, and attach thumbnails or diff screenshots for visual outputs when possible. Highlight any parameter changes (e.g., new `TOP_N`) and mention validation steps so reviewers can reproduce them.
