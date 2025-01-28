# Large Language Models for Genetic Disease Research and Diagnosis

This project presents a systematic review and analysis of how large language models (LLMs) can be applied in the research and diagnosis of human genetic diseases. It includes data collection, analysis pipelines, and the final dataset, which has been used to assess the role of LLMs in pre-analytical, analytical, and post-analytical diagnostic stages and education.

# For citation

If you want to cite this research, please use the following format:
`...`

# Github Structure

* `0_fetch_*.py` (+ utility script `utils_fetch.py`) - These scripts fetche unique articles from NCBI-PMC, BioRxiv, MedRxiv, and Arxiv, and stores them in the `data/` directory:
  - `data/0_ncbi.csv`
  - `data/0_bio_med.csv`
  - `data/0_arxiv_unique.csv`

* `1_merge_all.ipynb` - Merges all the data from the previous step into a single file `data/0_all.csv`, while removing duplicates. 

* `2_first_analysis.ipynb`:
  - Cleans the text data.
  - Merges rows that are similar in titles or abstracts, saving the output to `data/cleaned.csv`.
  - Analyzes word and phrase frequencies and produces histograms.
  - Extracts clinical-genetics related articles and saves them to `data/clinic_genetic.csv` (some additional manual deduplication was done afterwards).

* `3_old_new_merged.ipynb` -- needed if there are previous versions of syst reviews:
  - Gets all data from `old_data_dir`.
  - Annotate new data, if exist in old.
  - Export 2 files:
      - `new_data_file_out = 'data/clinic_genetic_formatted.csv'` - new data + old anno
      - `extra_data_file_out = 'data/clinic_genetic_extra.csv'` - old data, whic was not in new
* `4_highlight.py` -- to easily go through new data, and find keywords

* `data/` - Directory containing all the collected and processed data.

All further annotation was conducted manually using Google Spreadsheets.

# Data

The following are the primary data files generated and used in this project:

- `data/0_ncbi.csv` - Unique articles from NCBI-PMC.
- `data/0_bio_med.csv` - Unique articles from BioRxiv and MedRxiv.
- `data/0_arxiv_unique.csv` - Unique articles from Arxiv.
- `data/0_all.csv` - Combined data from all sources after removing duplicates.
- `data/cleaned.csv` - Cleaned and merged data.
- `data/clinic_genetic.csv` - Clinical-genetic articles selected for final analysis.
