#!/usr/bin/env python
# coding: utf-8

import logging
import os
from time import sleep

import arxiv
import pandas as pd
from tqdm import tqdm

from utils_fetch import start_date, end_date, query_terms_list, year_list, DATA_DIR
import os

# Configure the logging format and level
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def create_search_query(single_term, other_terms):
    # Helper function to format the query terms, adding quotes around multi-word terms
    def format_terms(terms):
        return " OR ".join([f"\"{term}\"" if " " in term else term for term in terms])

    # Format the second set of terms with OR between them
    query_part_2 = format_terms(other_terms)

    # Combine the single term from the first list with AND and the second list of terms
    final_query = f"(\"{single_term}\") AND ({query_part_2})"
    
    return final_query

# Function to fetch arXiv papers
def fetch_arxiv_papers(query, year_list=year_list):
    logging.info('Arxiv searching...')
    search = arxiv.Search(
        query=query,
        max_results=10000,  # arXiv has a limit of 1000 per query
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    i = 0
    for result in tqdm(search.results()):
        if result.published.year in year_list:
            results.append({
                'title': result.title,
                'abstract': result.summary,
                'url': result.entry_id,
                'published': result.published,
                'source': 'arXiv'
            })
        i+=1
        if i%100==0:
            sleep(20)
    return results

def save_collected(all_papers, name):
    pd.DataFrame(all_papers).to_csv(name, sep=',')

# Main loop to fetch data for each term in the first list
def fetch_and_merge_articles(query_terms_1, query_terms_2, year_list=year_list):
    all_results = []

    for term_1 in query_terms_1:
        try:
            query = create_search_query(term_1, query_terms_2)
            logging.info(f"Searching for: {query}")

            # Fetch articles for this query
            current_results = fetch_arxiv_papers(query, year_list=year_list)

            # Append the results
            all_results.extend(current_results)

            # Save progress after each term
            save_collected(all_results, os.path.join(DATA_DIR, f'0_arxiv_{term_1}.csv'))
            sleep(30)
        except arxiv.UnexpectedEmptyPageError as e:
            for term_2 in query_terms_2:
                sleep(30)
                query = create_search_query(term_1, [term_2,])
                logging.info(f"Searching for: {query}")

                # Fetch articles for this query
                current_results = fetch_arxiv_papers(query, year_list=year_list)

                # Append the results
                all_results.extend(current_results)

                # Save progress after each term
                save_collected(all_results, os.path.join(DATA_DIR, f'0_arxiv_{term_1}_{term_2}.csv'))

    return all_results

def remove_duplicates(articles):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(articles)
    
    # Drop duplicates based on title, abstract, and url (which should be unique)
    df_unique = df.drop_duplicates(subset=['title', 'abstract', 'url'])
    
    return df_unique



# In[7]:


logging.info("Starting article collection...")
all_arxiv_papers = fetch_and_merge_articles(query_terms_list[0], query_terms_list[1], year_list=year_list)
logging.info(f"Finished fetching papers, total found: {len(all_arxiv_papers)}")

# Remove duplicates
unique_papers = remove_duplicates(all_arxiv_papers)
logging.info(f"Total unique papers: {len(unique_papers)}")

# Save the final result
save_collected(unique_papers.to_dict('records'), os.path.join(DATA_DIR, '0_arxiv.csv'))

