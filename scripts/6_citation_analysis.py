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
CSV_FILE = '../data/final/ST2.csv'
TEX_FILE = '../data/manuscript/templateArxiv.tex'
OUTPUT_TABLE = '../results/citations/results_table.csv'
OUTPUT_PLOT = '../results/citations/results_plot.pdf'
OUTPUT_PLOT_SVG = '../results/citations/results_plot.svg'
OUTPUT_FIG3 = '../figures/fig3.pdf'  # Main figure for manuscript
OUTPUT_DETAILED = '../results/citations/detailed_citations.csv'
OUTPUT_STATS = '../results/citations/statistics_summary.txt'

# Section to category mapping
SECTION_MAPPING = {
    'intro': {
        'name': 'Introduction',
        'latex_section': r'\\section\{Introduction\}',
        'end_section': r'\\subsection\{Results',
    },
    'KN': {
        'name': 'Knowledge Navigation',
        'latex_section': r'\\subsection\{Knowledge navigation\}',
        'end_section': r'\\subsection\{',
        'subcategories': {
            'NER/RE': 'named entity recognition &\nrelation extraction',
            'RD': 'relation discovery'
        }
    },
    'CDA': {
        'name': 'Clinical Data Analysis',
        'latex_section': r'\\subsection\{Clinical data analysis\}',
        'end_section': r'\\subsection\{',
        'subcategories': {
            'CDN': 'clinical data normalization',
            'CDP': 'clinical diagnosis prediction',
            'MDP': 'molecular diagnosis prediction',
            'COP': 'outcome prediction'
        }
    },
    'GDA': {
        'name': 'Genetic Data Analysis',
        'latex_section': r'\\subsection\{Genetic data analysis\}',
        'end_section': r'\\subsection\{',
        'subcategories': {
            'AVE': 'analysis of variant effects',
            'GVI': 'genetic variant interpretation',
            'PP': 'phenotype prediction'
        }
    },
    'COM': {
        'name': 'Communication',
        'latex_section': r'\\subsection\{Interaction with patients and medical professionals\}',
        'end_section': r'\\subsection\{',
        'subcategories': {
            'MPC': 'medical professional\ncommunication',
            'PCC': 'patient communication &\ncounselling'
        }
    },
    'areas': {
        'name': 'Related Research Areas',
        'latex_section': r'\\subsection\{Related research areas\}',
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
    """Create summary table grouped by category and subcategory with supercategory totals."""
    summary_data = []

    # 1. Introduction
    intro_df = df[df['final_category'] == 'intro']
    summary_data.append({
        'Super category': '',
        'Article section': 'Introduction',
        'Research/application area': 'review',
        'Articles in ST2': len(intro_df),
        'Articles used in review': len(intro_df[intro_df['is_cited']]),
        'Unique articles in supercategory': ''
    })

    # 2. Results (5 groups)
    results_categories = ['KN', 'CDA', 'GDA', 'COM', 'areas']

    for cat_key in results_categories:
        cat_info = SECTION_MAPPING[cat_key]
        cat_name = cat_info['name']

        # Get articles for this category - unique articles
        cat_df = df[df['final_category'].str.contains(cat_key, na=False, regex=False)]
        unique_articles_in_supercategory = len(cat_df['ref'].unique())

        if 'subcategories' in cat_info:
            # Has subcategories - create rows for each
            for subcat_key, subcat_name in cat_info['subcategories'].items():
                subcat_df = cat_df[cat_df['subcategory'].str.contains(subcat_key, na=False, regex=False)]

                summary_data.append({
                    'Super category': 'Results',
                    'Article section': cat_name,
                    'Research/application area': subcat_name,
                    'Articles in ST2': len(subcat_df),
                    'Articles used in review': len(subcat_df[subcat_df['is_cited']]),
                    'Unique articles in supercategory': unique_articles_in_supercategory
                })
        else:
            # No subcategories (e.g., areas)
            summary_data.append({
                'Super category': 'Results',
                'Article section': cat_name,
                'Research/application area': '',
                'Articles in ST2': len(cat_df),
                'Articles used in review': len(cat_df[cat_df['is_cited']]),
                'Unique articles in supercategory': unique_articles_in_supercategory
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
        unique_disc = len(disc_df['ref'].unique())

        # Count how many are cited
        disc_cited = len(disc_df[disc_df['is_cited']])

        summary_data.append({
            'Super category': 'Discussion',
            'Article section': '',
            'Research/application area': disc_name,
            'Articles in ST2': disc_total,
            'Articles used in review': disc_cited,
            'Unique articles in supercategory': unique_disc
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def calculate_relevance_stats(df):
    """Calculate relevance statistics for annotated and used articles."""
    # Filter articles that have category assigned (annotated)
    annotated_df = df[df['final_category'].notna() & (df['final_category'] != '')]

    # Get relevance stats for all annotated articles
    relevance_all = annotated_df['relevance'].value_counts().sort_index()

    # Get relevance stats for cited articles
    cited_df = annotated_df[annotated_df['is_cited']]
    relevance_cited = cited_df['relevance'].value_counts().sort_index()

    stats = {
        'total_annotated': len(annotated_df),
        'total_cited': len(cited_df),
        'relevance_0_annotated': relevance_all.get(0, 0),
        'relevance_1_annotated': relevance_all.get(1, 0),
        'relevance_2_annotated': relevance_all.get(2, 0),
        'relevance_0_cited': relevance_cited.get(0, 0),
        'relevance_1_cited': relevance_cited.get(1, 0),
        'relevance_2_cited': relevance_cited.get(2, 0),
    }

    return stats


def create_text_summary(df, summary_df, relevance_stats):
    """Create text summary with aggregated statistics."""
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("AGGREGATED STATISTICS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    # Overall statistics
    summary_lines.append("OVERALL STATISTICS:")
    summary_lines.append(f"  Total articles in ST2: {len(df)}")
    summary_lines.append(f"  Total annotated articles: {relevance_stats['total_annotated']}")
    summary_lines.append(f"  Total cited articles: {relevance_stats['total_cited']}")
    summary_lines.append(f"  Citation rate: {relevance_stats['total_cited']/relevance_stats['total_annotated']*100:.1f}%")
    summary_lines.append("")

    # Relevance distribution for annotated articles
    summary_lines.append("RELEVANCE DISTRIBUTION (ANNOTATED ARTICLES):")
    summary_lines.append(f"  Relevance 0 (irrelevant): {relevance_stats['relevance_0_annotated']} ({relevance_stats['relevance_0_annotated']/relevance_stats['total_annotated']*100:.1f}%)")
    summary_lines.append(f"  Relevance 1 (partially relevant): {relevance_stats['relevance_1_annotated']} ({relevance_stats['relevance_1_annotated']/relevance_stats['total_annotated']*100:.1f}%)")
    summary_lines.append(f"  Relevance 2 (fully relevant): {relevance_stats['relevance_2_annotated']} ({relevance_stats['relevance_2_annotated']/relevance_stats['total_annotated']*100:.1f}%)")
    summary_lines.append("")

    # Relevance distribution for cited articles
    summary_lines.append("RELEVANCE DISTRIBUTION (CITED ARTICLES):")
    summary_lines.append(f"  Relevance 0 (irrelevant): {relevance_stats['relevance_0_cited']} ({relevance_stats['relevance_0_cited']/relevance_stats['total_cited']*100 if relevance_stats['total_cited'] > 0 else 0:.1f}%)")
    summary_lines.append(f"  Relevance 1 (partially relevant): {relevance_stats['relevance_1_cited']} ({relevance_stats['relevance_1_cited']/relevance_stats['total_cited']*100 if relevance_stats['total_cited'] > 0 else 0:.1f}%)")
    summary_lines.append(f"  Relevance 2 (fully relevant): {relevance_stats['relevance_2_cited']} ({relevance_stats['relevance_2_cited']/relevance_stats['total_cited']*100 if relevance_stats['total_cited'] > 0 else 0:.1f}%)")
    summary_lines.append("")

    # Supercategory statistics (Results sections only)
    summary_lines.append("SUPERCATEGORY STATISTICS (RESULTS SECTIONS):")
    results_df = summary_df[summary_df['Super category'] == 'Results']

    for cat_name in results_df['Article section'].unique():
        cat_rows = results_df[results_df['Article section'] == cat_name]
        unique_count = cat_rows['Unique articles in supercategory'].iloc[0]
        total_in_subcats = cat_rows['Articles in ST2'].sum()
        cited_in_subcats = cat_rows['Articles used in review'].sum()

        summary_lines.append(f"  {cat_name}:")
        summary_lines.append(f"    Unique articles: {unique_count}")
        summary_lines.append(f"    Total entries (with duplicates): {total_in_subcats}")
        summary_lines.append(f"    Cited articles: {cited_in_subcats}")

    summary_lines.append("")
    summary_lines.append("=" * 70)

    return "\n".join(summary_lines)


def create_visualization(summary_df, output_file):
    """Create horizontal bar chart showing only Results sections (KN, CDA, GDA, COM, areas)."""
    # Set Helvetica font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

    # Filter only Results sections (exclude Introduction and Discussion)
    plot_df = summary_df[summary_df['Super category'] == 'Results'].copy()

    # Convert ST2 column to numeric (should already be numbers now)
    plot_df['Articles in ST2'] = pd.to_numeric(plot_df['Articles in ST2'], errors='coerce').fillna(0)

    # Define color scheme for each supercategory
    # Format: {section_name: (fill_color, edge_color)}
    color_scheme = {
        'Knowledge Navigation': ('#fff2cc', '#ddda46'),
        'Clinical Data Analysis': ('#fad7ac', '#b46504'),
        'Genetic Data Analysis': ('#fad9d5', '#ae4132'),
        'Communication': ('#bac8d3', '#23445d'),
        'Related Research Areas': ('#d3d3d3', '#696969')
    }

    # Capitalize research area names with custom rules and add line breaks
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

    def add_line_breaks(text):
        """Add line breaks to long labels for compactness."""
        if not text:
            return text

        # Specific replacements for long labels
        replacements = {
            'Named Entity Recognition & Relation Extraction': 'Named Entity Recognition &\nRelation Extraction',
            'Medical Professional Communication': 'Medical Professional\nCommunication',
            'Patient Communication & Counselling': 'Patient Communication &\nCounselling',
            'Clinical Data Normalization': 'Clinical Data\nNormalization',
            'Clinical Diagnosis Prediction': 'Clinical Diagnosis\nPrediction',
            'Molecular Diagnosis Prediction': 'Molecular Diagnosis\nPrediction',
            'Analysis of Variant Effects': 'Analysis of\nVariant Effects',
            'Genetic Variant Interpretation': 'Genetic Variant\nInterpretation',
            'Related Research Areas': 'Related Research\nAreas'
        }

        return replacements.get(text, text)

    plot_df['Research/application area'] = plot_df['Research/application area'].apply(smart_title).apply(add_line_breaks)
    plot_df['Article section'] = plot_df['Article section'].apply(smart_title)

    plot_df = plot_df.reset_index(drop=True)

    # Reverse for plotting (top to bottom)
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    # Calculate height
    num_rows = len(plot_df)
    fig_height = max(10, num_rows * 0.5)

    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(plot_df))
    height = 0.6

    # Create horizontal bars with different colors per supercategory
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        section = row['Article section']
        value = row['Articles in ST2']

        # Get colors for this section
        fill_color, edge_color = color_scheme.get(section, ('#d3d3d3', '#696969'))

        ax.barh(i, value, height, alpha=0.9,
                color=fill_color, edgecolor=edge_color, linewidth=1.5)

    # Y-axis labels - use research area for subcategories, article section for standalone
    y_labels = []
    for idx, row in plot_df.iterrows():
        research_area = row['Research/application area']
        article_section = row['Article section']

        # If research area is empty (standalone section like "areas"), use article section
        if research_area == '' or article_section in ['Related Research Areas']:
            y_labels.append(article_section)
        else:
            y_labels.append(research_area)

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=int(13 * 1.4))

    # Styling with increased font sizes (x1.4)
    ax.set_xlabel('Number of Articles', fontsize=int(16 * 1.4), fontweight='bold')
    ax.set_title('Final Set of Articles in Each Section',
                 fontsize=int(18 * 1.4), fontweight='bold', pad=20)
    # Remove legend as requested
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels at the end of bars with increased font size
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        st2_val = row['Articles in ST2']

        # Label for bar - only if not zero
        if st2_val > 0:
            ax.text(st2_val + 0.5, i, str(int(st2_val)),
                    va='center', ha='left', fontsize=int(12 * 1.4), fontweight='bold', color='black')

    # Add section separators and labels based on Article section
    plot_df_original = plot_df.iloc[::-1].reset_index(drop=True)

    current_key = None
    section_ranges = []
    start_idx = 0

    for idx, row in plot_df_original.iterrows():
        section = row['Article section']

        # Use article section as grouping key
        if section in ['Related Research Areas']:
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
    max_val = plot_df['Articles in ST2'].max()

    for i, section_range in enumerate(section_ranges):
        start = len(plot_df) - 1 - section_range['end']
        end = len(plot_df) - 1 - section_range['start']
        mid_pos = (start + end) / 2

        # Draw separator line
        if i > 0:
            ax.axhline(y=end + 0.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.6)

        # Add section label with increased font size (x1.4)
        display_name = section_range['display_name']
        if display_name:
            ax.text(max_val * 1.15, mid_pos, display_name,
                    fontsize=int(13 * 1.4), fontweight='bold',
                    va='center', ha='left', rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Also save as SVG
    svg_file = output_file.replace('.pdf', '.svg')
    plt.savefig(svg_file, format='svg', bbox_inches='tight')

    plt.close()
    print(f"\nVisualization saved to {output_file}")
    print(f"SVG version saved to {svg_file}")


def main():
    print("=" * 60)
    print("Analyzing Citations in LaTeX Manuscript")
    print("=" * 60)

    # Step 1: Parse LaTeX
    print("\n[1/7] Parsing LaTeX file...")
    section_citations = parse_latex_citations(TEX_FILE)
    total_citations = sum(len(cites) for cites in section_citations.values())
    print(f"Total unique citations found: {total_citations}")

    # Step 2: Load CSV
    print("\n[2/7] Loading CSV file...")
    df = process_csv(CSV_FILE)
    print(f"Total articles in ST2: {len(df)}")

    # Step 3: Analyze
    print("\n[3/7] Analyzing which articles are cited...")
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
    print("\n[4/7] Creating summary table...")
    summary_df = create_summary_table(df)
    summary_df.to_csv(OUTPUT_TABLE, index=False)
    print(f"Summary table saved to {OUTPUT_TABLE}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    # Step 5: Relevance statistics
    print("\n[5/7] Calculating relevance statistics...")
    relevance_stats = calculate_relevance_stats(df)

    # Step 6: Text summary
    print("\n[6/7] Creating text summary...")
    text_summary = create_text_summary(df, summary_df, relevance_stats)
    with open('./statistics_summary.txt', 'w', encoding='utf-8') as f:
        f.write(text_summary)
    print(f"Text summary saved to ./statistics_summary.txt")
    print("\n" + text_summary)

    # Step 7: Visualization
    print("\n[7/7] Creating visualization...")
    create_visualization(summary_df, OUTPUT_PLOT)

    # Copy to figures directory as fig3.pdf for manuscript
    import shutil
    shutil.copy2(OUTPUT_PLOT, OUTPUT_FIG3)
    print(f"Figure also saved to {OUTPUT_FIG3}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
