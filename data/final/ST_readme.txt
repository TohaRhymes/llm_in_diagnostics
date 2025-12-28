### Column Descriptions

| Column              | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `title`             | Title of the article                                                        |
| `abstract`          | Abstract of the article                                                     |
| `source`            | Source(s) where the article was found; duplicates may have multiple sources |
| `review`            | Indicates review article: `1` = review, `0` = not review, empty = unknown   |
| `relevance`         | Overall relevance of the article: 1,2,3 (0 - irrelevant, 1 - partially, 2 - relevant)|
| `code`              | (Reserved / unused)                                                         |
| `what section used` | Main section(s) where the article is used (see mapping below)               |
| `subgroup`          | Specific task within the section (see article for details)                  |
| `ref`               | Reference ID (for BibTeX or internal lookup)                                |
| `ai_topic`          | AI-related notes (optional)                                                 |
| `medicine_topic`    | Medicine-related notes (optional)                                           |
| `notes`             | General notes                                                               |
| `not_relevant`      | Boolean flag: article is irrelevant (`TRUE` / `FALSE`)                      |
| `partly_relevant`   | Boolean flag: article is partially relevant                                 |
| `relevant`          | Boolean flag: article is relevant                                           |

---

### `what section used` Mapping

| Code          | Phase          | Section Name                                      |
| ------------- | -------------- | ------------------------------------------------- |
| `intro: rev`  | Introduction   | Review                                            |
| `pre: KNLR`   | Pre-Analytics  | Knowledge Navigation & Literature Review          |
| `pre: RS`     | Pre-Analytics  | Risk Stratification                               |
| `ana: MIA`    | Analytics      | Medical Imaging Analysis                          |
| `ana: AVE`    | Analytics      | Analysis of Variant Effects                       |
| `ana: CVI`    | Analytics      | Clinical Variant Interpretation                   |
| `post: PCS`   | Post-Analytics | Patient Clustering & Concept Typing               |
| `post: DRA`   | Post-Analytics | Data & Results Aggregation                        |
| `post: CRGDS` | Post-Analytics | Clinical Report Generation & Decision Support     |
| `edu`         | Education      | Educational Use                                   |
| `disc`        | Discussion     | General LLM Use or Non-categorized Medical Topics |

---

### `subgroup` Notes

The `subgroup` field provides fine-grained task labels within a section (e.g., "Named Entity Recognition", "Phenotype Extraction", "Variant Prioritization").
Refer to the article directly for subgroup meaning and examples.
