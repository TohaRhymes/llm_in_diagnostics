#!/usr/bin/env python
# coding: utf-8

import json
import logging

from Bio import Entrez
import pandas as pd
from tqdm import tqdm

from utils_fetch import start_date, end_date, query_terms_list

# Configure the logging format and level
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


# In[ ]:


# Configure Entrez for NCBI API
Entrez.email = "anton@gmail.com"  # Replace with your email

def create_search_query(query_terms):
    # Helper function to format the query terms, adding quotes around multi-word terms
    def format_terms(terms):
        return " OR ".join([f"\"{term}\"" if " " in term else term for term in terms])

    # Format the two sets of terms with OR between them
    query_part_1 = format_terms(query_terms[0])
    query_part_2 = format_terms(query_terms[1])

    # Combine the two parts with AND between them
    final_query = f"({query_part_1}) AND ({query_part_2})"
    
    return final_query

# Function to fetch NCBI (PubMed) papers
def fetch_ncbi_papers(query):
    logging.info('ncbi searching...')
    handle = Entrez.esearch(db="pubmed", 
                            term=query, 
                            retmax=1000000, 
                            mindate="2023", 
                            maxdate="2024")
    record = Entrez.read(handle)
    id_list = record["IdList"]
    handle.close()

    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
    records = Entrez.read(handle)
    results = []
    for article in tqdm(records['PubmedArticle']):
        title = article['MedlineCitation']['Article']['ArticleTitle']
        abstract = ""
        try:
            abstract = " ".join(article['MedlineCitation']['Article']['Abstract']['AbstractText'])
        except KeyError:
            pass
        
        # Extract publication date
        published = ""
        try:
            pub_date = article['MedlineCitation']['Article']['ArticleDate'][0]
            published = f"{pub_date['Year']}-{pub_date['Month']}-{pub_date['Day']}" if pub_date else "Unknown"
        except IndexError:
            pass

        # Extract DOI (to construct the URL)
        doi = None
        for id_tag in article['PubmedData']['ArticleIdList']:
            if id_tag.attributes['IdType'] == 'doi':
                doi = str(id_tag)
                break
        url = f"https://doi.org/{doi}" if doi else "No DOI"

        results.append({
            'title': title,
            'abstract': abstract,
            'url': url,
            'published': published,
            'source': 'PubMed'
        })
    return results

def save_collected(all_papers, name):
    pd.DataFrame(all_papers).to_csv(name, sep=',')


# In[7]:


query = create_search_query(query_terms_list)

# now actually search
logging.info("JUST STARTED")

ncbi_papers = fetch_ncbi_papers(query)
logging.info(f"FINISHED ncbi, len: {len(ncbi_papers)}")
save_collected(ncbi_papers, 'data/0_ncbi.csv')


