#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

from utils_fetch import start_date, end_date, query_terms_list, year_list, DATA_DIR


# Define the function to highlight a word in the text
def highlight_word(text, word,color):
    return text.replace(word, f'{color}{word}\033[0m')  # Using ANSI escape code for yellow


def flatten_list(nested_list):
    """
    Flattens a list of lists into a single list.
    
    :param nested_list: List of lists to be flattened
    :return: A single flattened list
    """
    return [item for sublist in nested_list for item in sublist]


DATA_DIR = 'data'

phrases_to_include_list = [
    [
        "LLM",
        "large language model",
         "NLP", 
         "natural language processing",
        "GPT",
        "chatGPT",
        "transformer",
        "BERT",
        "Bidirectional Encoder Representation",
        "RAG",
        "augmented generation",
        "generative AI",
        "AI assistant",
        "prompt engineering",
        "chatbot",
        "prompt engineering",
        "attention mechanism",
        "chain-of-thought",
        "chain of thought",
    ],
    [
        "electronic health record",
        "ehr",
        "clinical",
        "case report",
        "cds",
        "intensive care unit",
        "medical",
        "syndrome",
        "phenotype",
        "complex trait",
    ],
    [
        "inherit",
        "heredit",
        "heritability",
        "gwas",
        "genome-wide",
        "genome wide",
        "association stud",
        "snp",
        "single nucleotide",
        "genetic",
        "variant interpretation",
        "genomic varia",
        "human gen",
        "NGS",
        "generation sequencing",
    ],
]

colors=["\033[93m", "\033[91m", "\033[92m"]

phrases_to_check = flatten_list(phrases_to_include_list)
phrases_list_lower = [[i.lower() for i in j] for j in phrases_to_include_list]

file_to_highlight = os.path.join(DATA_DIR, 'clinic_genetic.csv') 
file_to_highlight = os.path.join(DATA_DIR, '_clinic_genetic_formatted_ST1_raw.csv') 

data = pd.read_csv(file_to_highlight)[['title', 'abstract', 'source']]
shape = data.shape[0]
for index_of_absract, (title, abstract) in enumerate(zip(data.title, data.abstract)):    
    
    found_phrases = [w for w in phrases_to_check if w.lower() in str(abstract).lower() or w.lower() in str(title).lower()]
    
    colors_list = []
    for phrase in found_phrases:
        for i, ph_l in enumerate(phrases_list_lower):
            if phrase.lower() in ph_l:
                colors_list.append(colors[i])
    print(len(found_phrases), len(colors_list))
    
    # Highlight each found phrase in the abstract
    highlighted_abstract = str(abstract).lower()
    highlighted_title = str(title).lower()
    if found_phrases or True:
        for phrase, color in zip(found_phrases, colors_list):
            highlighted_abstract = highlight_word(highlighted_abstract, phrase.lower(), color)
            highlighted_title = highlight_word(highlighted_title, phrase.lower(), color)
        print(f'ARTICLE {index_of_absract}/{shape}')
        print(title)
        print(highlighted_title)
        print(highlighted_abstract)
        print(found_phrases)
        print('---------------------')
