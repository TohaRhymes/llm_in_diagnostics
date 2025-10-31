#!/usr/bin/env python3
"""
Script to analyze which articles from ST2.csv are cited in the LaTeX manuscript.
Creates a table and visualization comparing total articles vs. cited articles.
"""

import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
CSV_FILE = './data/ST2_v2.csv'
TEX_FILE = './data/templateArxiv.tex'
OUTPUT_TABLE = './results_table.csv'
OUTPUT_PLOT = './results_plot.pdf'

# Section to category mapping
SECTION_MAPPING = {
    'intro': {
        'name': 'Introduction',
        'latex_section': r'\\section\{Introduction\}',
        'end_section': r'\\subsection\{Results',
    },
    'KN': {
        'name': 'Knowledge Navigation',
        'latex_section': r'\\subsubsection\{Knowledge navigation\}',
        'end_section': r'\\subsubsection\{',
        'subcategories': {
            'NER/RE': 'named entity recognition &\nrelation extraction',
            'RD': 'relation'
        }
    },
    'CDA': {
        'name': 'Clinical Data Analysis',
        'latex_section': r'\\subsubsection\{Clinical data analysis\}',
        'end_section': r'\\subsubsection\{',
        'subcategories': {
            'CDN': 'clinical data normalization',
            'CDP': 'clinical diagnosis prediction',
            'MDP': 'molecular diagnosis prediction',
            'COP': 'outcome prediction'
        }
    },
    'GDA': {
        'name': 'Genetic Data Analysis',
        'latex_section': r'\\subsubsection\{Genetic data analysis\}',
        'end_section': r'\\subsubsection\{',
        'subcategories': {
            'AVE': 'analysis of variant effects',
            'GVI': 'genetic variant interpretation',
            'PP': 'phenotype prediction'
        }
    },
    'COM': {
        'name': 'Communication',
        'latex_section': r'\\subsubsection\{Interaction with patients and medical professionals\}',
        'end_section': r'\\subsubsection\{',
        'subcategories': {
            'MPC': 'medical professional\ncommunication',
            'PC': 'patient communication &\ncounselling'
        }
    },
    'areas': {
        'name': 'Related Research Areas',
        'latex_section': r'\\subsubsection\{Related research areas\}',
        'end_section': r'\\section\{Discussion\}',
    },
    'techniques': {
        'name': 'LLMs Selection Guide & Model Strategies',
        'latex_section': r'\\subsection\{LLMs selection guide\}',
        'end_section': r'\\subsection\{Data and Benchmarks\}',
    },
    'data': {
        'name': 'Data and Benchmarks',
        'latex_section': r'\\subsection\{Data and Benchmarks\}',
        'end_section': r'\\subsection\{Biases\}',
    },
    'biases': {
        'name': 'Biases',
        'latex_section': r'\\subsection\{Biases\}',
        'end_section': r'\\section\{',
    },
}


def extract_section_text(tex_content, start_pattern, end_pattern):
    """Extract text between two section markers."""
    start_match = re.search(start_pattern, tex_content, re.IGNORECASE)
    if not start_match:
        return ""

    start_pos = start_match.start()

    # Find the next section after start
    end_matches = list(re.finditer(end_pattern, tex_content[start_pos + len(start_match.group()):], re.IGNORECASE))

    if end_matches:
        end_pos = start_pos + len(start_match.group()) + end_matches[0].start()
    else:
        end_pos = len(tex_content)

    return tex_content[start_pos:end_pos]


def extract_citations_from_text(text):
    """Extract all citation keys from a text block."""
    citations = set()
    cite_pattern = r'\\cite\{([^}]+)\}'
    matches = re.findall(cite_pattern, text)

    for match in matches:
        keys = [k.strip() for k in match.split(',')]
        citations.update(keys)

    return citations


def parse_latex_citations(tex_file):
    """Parse LaTeX file and extract citations by section."""
    with open(tex_file, 'r', encoding='utf-8') as f:
        tex_content = f.read()

    section_citations = {}

    for section_key, section_info in SECTION_MAPPING.items():
        section_text = extract_section_text(
            tex_content,
            section_info['latex_section'],
            section_info['end_section']
        )
        citations = extract_citations_from_text(section_text)
        section_citations[section_key] = citations
        print(f"{section_key}: {len(citations)} citations found")

    return section_citations


def process_csv(csv_file):
    """Process the ST2 CSV file."""
    df = pd.read_csv(csv_file)
    df['ref'] = df['ref'].astype(str).str.strip()
    return df


def analyze_citations(df, section_citations):
    """Analyze which articles from ST2 are cited in which sections."""
    df['cited_in_sections'] = ''
    df['is_cited'] = False

    # Get set of all refs in ST2 for filtering
    st2_refs = set(df['ref'].astype(str).str.strip())

    for idx, row in df.iterrows():
        ref_key = str(row['ref']).strip()
        cited_sections = []

        # Check each section for this citation
        for section_key, citations in section_citations.items():
            if ref_key in citations:
                cited_sections.append(section_key)

        if cited_sections:
            df.at[idx, 'cited_in_sections'] = ', '.join(cited_sections)
            df.at[idx, 'is_cited'] = True

    # Also track which citations in each section are actually in ST2
    section_citations_filtered = {}
    for section_key, citations in section_citations.items():
        # Only keep citations that are in ST2
        section_citations_filtered[section_key] = citations & st2_refs

    return df, section_citations_filtered


def create_summary_table(df):
    """Create summary table grouped by category and subcategory."""
    summary_data = []

    # 1. Introduction
    intro_df = df[df['final_category'] == 'intro']
    summary_data.append({
        'Super category': '',
        'Article section': 'Introduction',
        'Research/application area': 'review',
        'Articles in ST2': len(intro_df),
        'Articles used in review': len(intro_df[intro_df['is_cited']])
    })

    # 2. Results (5 groups)
    results_categories = ['KN', 'CDA', 'GDA', 'COM', 'areas']

    for cat_key in results_categories:
        cat_info = SECTION_MAPPING[cat_key]
        cat_name = cat_info['name']

        # Get articles for this category
        cat_df = df[df['final_category'].str.contains(cat_key, na=False, regex=False)]

        if 'subcategories' in cat_info:
            # Has subcategories - create rows for each
            for subcat_key, subcat_name in cat_info['subcategories'].items():
                subcat_df = cat_df[cat_df['subcategory'].str.contains(subcat_key, na=False, regex=False)]

                summary_data.append({
                    'Super category': 'Results',
                    'Article section': cat_name,
                    'Research/application area': subcat_name,
                    'Articles in ST2': len(subcat_df),
                    'Articles used in review': len(subcat_df[subcat_df['is_cited']])
                })
        else:
            # No subcategories (e.g., areas)
            summary_data.append({
                'Super category': 'Results',
                'Article section': cat_name,
                'Research/application area': '',
                'Articles in ST2': len(cat_df),
                'Articles used in review': len(cat_df[cat_df['is_cited']])
            })

    # 3. Discussion (3 groups)
    discussion_categories = {
        'techniques': 'LLMs selection guide & model strategies',
        'data': 'data and benchmarks',
        'biases': 'biases and limitations'
    }

    for disc_key, disc_name in discussion_categories.items():
        # Count articles in ST2 with this category
        disc_df = df[df['final_category'].str.contains(disc_key, na=False, regex=False)]
        disc_total = len(disc_df)

        # Count how many are cited
        disc_cited = len(disc_df[disc_df['is_cited']])

        summary_data.append({
            'Super category': 'Discussion',
            'Article section': '',
            'Research/application area': disc_name,
            'Articles in ST2': disc_total,
            'Articles used in review': disc_cited
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def create_visualization(summary_df, output_file):
    """Create horizontal grouped bar chart with category separators."""
    # Include ALL rows (including Discussion)
    plot_df = summary_df.copy()

    # Convert ST2 column to numeric (should already be numbers now)
    plot_df['Articles in ST2'] = pd.to_numeric(plot_df['Articles in ST2'], errors='coerce').fillna(0)

    # Capitalize research area names with custom rules
    def smart_title(text):
        """Apply title case but keep certain words lowercase and fix abbreviations."""
        if not text:
            return text

        # Words that should be lowercase
        lowercase_words = {'and', 'or', 'of', 'the', 'in', 'with', 'for', 'to', 'a', 'an'}

        # Special cases for abbreviations
        special_cases = {
            'llms': 'LLMs',
            'llm': 'LLM',
            'ai': 'AI',
        }

        words = text.split()
        result = []

        for i, word in enumerate(words):
            word_lower = word.lower()

            # Check special cases first
            if word_lower in special_cases:
                result.append(special_cases[word_lower])
            # First word or not a lowercase word
            elif i == 0 or word_lower not in lowercase_words:
                result.append(word.capitalize())
            else:
                result.append(word_lower)

        return ' '.join(result)

    plot_df['Research/application area'] = plot_df['Research/application area'].apply(smart_title)
    plot_df['Article section'] = plot_df['Article section'].apply(smart_title)

    plot_df = plot_df.reset_index(drop=True)

    # Reverse for plotting (top to bottom)
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    # Calculate height
    num_rows = len(plot_df)
    fig_height = max(10, num_rows * 0.5)

    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(plot_df))
    height = 0.35

    # Create horizontal bars with black borders (matching Petr's style)
    ax.barh(y - height/2, plot_df['Articles in ST2'],
            height, label='Articles after reviewing', alpha=0.8,
            color='steelblue', edgecolor='black', linewidth=1)
    ax.barh(y + height/2, plot_df['Articles used in review'],
            height, label='Articles cited in manuscript', alpha=0.8,
            color='coral', edgecolor='black', linewidth=1)

    # Y-axis labels - use Article section for single-item categories, otherwise use research area
    y_labels = []
    for idx, row in plot_df.iterrows():
        research_area = row['Research/application area']
        article_section = row['Article section']
        super_cat = row['Super category']

        # If research area is empty or this is a standalone section, use article section
        if research_area == '' or article_section in ['Related Research Areas', 'Introduction']:
            y_labels.append(article_section)
        # For Discussion items, use the research area
        elif super_cat == 'Discussion':
            y_labels.append(research_area)
        else:
            y_labels.append(research_area)

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=11)

    # Styling
    ax.set_xlabel('Number of Articles', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Total Articles vs. Articles Cited in Review',
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper center', fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels at the end of bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        st2_val = row['Articles in ST2']
        used_val = row['Articles used in review']

        # Label for ST2 (blue bar) - only if not zero
        if st2_val > 0:
            ax.text(st2_val + 0.5, i - height/2, str(int(st2_val)),
                    va='center', ha='left', fontsize=9, fontweight='bold', color='black')

        # Label for used articles (orange bar)
        ax.text(used_val + 0.5, i + height/2, str(int(used_val)),
                va='center', ha='left', fontsize=9, fontweight='bold', color='black')

    # Add section separators and labels based on Article section OR Super category
    plot_df_original = plot_df.iloc[::-1].reset_index(drop=True)

    current_key = None
    section_ranges = []
    start_idx = 0

    for idx, row in plot_df_original.iterrows():
        section = row['Article section']
        super_cat = row['Super category']

        # Use super_cat for Discussion, otherwise use section
        if super_cat == 'Discussion':
            group_key = 'Discussion'
            display_name = 'Discussion'
        elif section in ['Related Research Areas', 'Introduction']:
            group_key = section
            display_name = section
        else:
            group_key = section
            display_name = section

        if group_key != current_key:
            if current_key is not None:
                # Save previous section range
                section_ranges.append({
                    'start': start_idx,
                    'end': idx - 1,
                    'display_name': current_display
                })
            start_idx = idx
            current_key = group_key
            current_display = display_name

    # Add last section
    if current_key is not None:
        section_ranges.append({
            'start': start_idx,
            'end': len(plot_df_original) - 1,
            'display_name': current_display
        })

    # Draw separators and labels
    max_val = max(plot_df['Articles in ST2'].max(), plot_df['Articles used in review'].max())

    for i, section_range in enumerate(section_ranges):
        start = len(plot_df) - 1 - section_range['end']
        end = len(plot_df) - 1 - section_range['start']
        mid_pos = (start + end) / 2

        # Draw separator line
        if i > 0:
            ax.axhline(y=end + 0.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.6)

        # Add section label (without box)
        display_name = section_range['display_name']
        if display_name and display_name != 'Introduction':
            ax.text(max_val * 1.15, mid_pos, display_name,
                    fontsize=11, fontweight='bold',
                    va='center', ha='left', rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {output_file}")


def main():
    print("=" * 60)
    print("Analyzing Citations in LaTeX Manuscript")
    print("=" * 60)

    # Step 1: Parse LaTeX
    print("\n[1/5] Parsing LaTeX file...")
    section_citations = parse_latex_citations(TEX_FILE)
    total_citations = sum(len(cites) for cites in section_citations.values())
    print(f"Total unique citations found: {total_citations}")

    # Step 2: Load CSV
    print("\n[2/5] Loading CSV file...")
    df = process_csv(CSV_FILE)
    print(f"Total articles in ST2: {len(df)}")

    # Step 3: Analyze
    print("\n[3/5] Analyzing which articles are cited...")
    df, section_citations_filtered = analyze_citations(df, section_citations)
    cited_count = df['is_cited'].sum()
    print(f"Articles cited in manuscript: {cited_count} ({cited_count/len(df)*100:.1f}%)")

    # Print filtered citation counts
    print("\nCitations in ST2 per section:")
    for section_key, citations in section_citations_filtered.items():
        print(f"  {section_key}: {len(citations)} articles from ST2")

    df.to_csv('./detailed_citations.csv', index=False)
    print(f"\nDetailed results saved to ./detailed_citations.csv")

    # Step 4: Summary table
    print("\n[4/5] Creating summary table...")
    summary_df = create_summary_table(df)
    summary_df.to_csv(OUTPUT_TABLE, index=False)
    print(f"Summary table saved to {OUTPUT_TABLE}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    # Step 5: Visualization
    print("\n[5/5] Creating visualization...")
    create_visualization(summary_df, OUTPUT_PLOT)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
