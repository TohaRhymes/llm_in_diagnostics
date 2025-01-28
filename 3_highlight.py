#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

from utils_fetch import start_date, end_date, query_terms_list, year_list, DATA_DIR


# Define the function to highlight a word in the text
def highlight_word(text, word):
    return text.replace(word, f'\033[93m{word}\033[0m')  # Using ANSI escape code for yellow


def flatten_list(nested_list):
    """
    Flattens a list of lists into a single list.
    
    :param nested_list: List of lists to be flattened
    :return: A single flattened list
    """
    return [item for sublist in nested_list for item in sublist]


DATA_DIR = 'data_v2'

phrases_to_include_list = [
    [
        "varchat"
#         "LLM", 
#      "language model", 
#      "NLP", 
#      "natural language processing",
#      "GPT", 
#      "chatGPT", 
#      "transformer", 
#      "BERT", 
#      "Bidirectional Encoder Representation", 
#      "RAG", 
#      "augmented generation", 
#      "generative AI", 
#      "AI assistant", 
#      "prompt", 
#      "chatbot", 
#      "prompt engineering", 
#      "attention mechanism", 
#      "chain-of-thought", 
#      "chain of thought", 
#      "agent"
#     ],
#     ['electronic health record', 
#      'ehr', 
#      'clinical', 
#      'case report',
#      'cds', 
#      "intensive care unit",
#      'medical', 
#      'syndrome', 
#      'phenotype', 
#      "complex trait"],
#     ["inherit", 
#      "heredit", 
#      "heritability", 
#      "gwas", 
#      "genome-wide", 
#      "genome wide", 
#      "association stud", 
#      "snp",
#      "single nucleotide",
#      "genetic", 
#      "variant interpretation", 
#      "genomic varia", 
#      "human gen",
#      "NGS",
#      "generation sequencing"
    ]
]

phrases_to_check = flatten_list(phrases_to_include_list)


data = pd.read_csv(os.path.join(DATA_DIR, '0_all.csv'))[['title', 'abstract', 'source']]

for title, abstract in zip(data.title, data.abstract):
    
    found_phrases = [w for w in phrases_to_check if w.lower() in str(abstract).lower()]
    
    # Highlight each found phrase in the abstract
    highlighted_abstract = str(abstract).lower()
    if found_phrases:
        for phrase in found_phrases:
            print(title)
            highlighted_abstract = highlight_word(highlighted_abstract, phrase.lower())

        print(highlighted_abstract)
        print(found_phrases)