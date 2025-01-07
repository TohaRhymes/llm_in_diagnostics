#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import logging

import arxiv
import requests
import pandas as pd
from tqdm import tqdm

from utils_fetch import start_date, end_date, query_terms_list

# Configure the logging format and level
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Function to fetch bioRxiv and medRxiv papers
# Function to fetch papers from bioRxiv or medRxiv
def fetch_papers_rxiv(server, 
                      start_date, 
                      end_date, 
                      query_terms_list):
    results = []
    cursor = 0
    total_papers = None
    new_papers_count = None
    while True:
        # Construct the API URL with the correct format, iterating over pages using the cursor
        api_url = f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/{cursor}/json"
        
        # Set headers to mimic a legitimate browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Make the API request
        response = requests.get(api_url, headers=headers)
        logging.info(f"server: {server}; cursor = {cursor}, total = {total_papers}, new = {new_papers_count}, added = {len(results)}")
        if response.status_code == 200:
            data = response.json()
            messages = data.get('messages', [{}])[0]  # Get the first message

            # Extract pagination metadata
            total_papers = messages.get('total', None)  # Total papers in the database
            new_papers_count = messages.get('count_new_papers', None)  # New papers in the range
            
            papers = data.get('collection', [])
            
            # Filter the papers based on the provided query terms
            for paper in papers:
                title = paper['title']
                abstract = paper['abstract']
                published_date = paper['date']
                doi = paper['doi']
                url = f"https://doi.org/{doi}"
                authors = paper['authors']
                flag = True
                for query_terms in query_terms_list: 
                # Check if any query term matches the title or abstract
                    if not any(term.lower() in title.lower() or term.lower() in abstract.lower() for term in query_terms):
                        flag = False
                if flag:
                    results.append({
                        'title': title,
                        'abstract': abstract,
                        'url': url,
                        'published': published_date,
                        'authors': authors,
                        'source': server
                    })
            
            # Pagination: Check if we should continue to the next page
            if len(papers) < 100:
                break  # Stop if there are fewer than 100 papers or no more cursor
            # Update the cursor for the next request
            cursor += 100  # Increment the cursor by 100 to get the next page
        else:
            logging.warning(f"Error {response.status_code}: {response.text}")
            break

    return results

# Function to loop through bioRxiv and medRxiv
def fetch_rxiv_both_servers(start_date, end_date, query_terms_list):
    all_results = []

    # Loop through both servers: 'biorxiv' and 'medrxiv'
    for server in ['biorxiv', 'medrxiv']:
        logging.info(f"Fetching papers from {server}...")
        server_results = fetch_papers_rxiv(server, start_date, end_date, query_terms_list)
        all_results.extend(server_results)

    return all_results


# In[3]:


def save_collected(all_papers, name):
    pd.DataFrame(all_papers).to_csv(name, sep=',')


# In[7]:


# now actually search
logging.info("JUST STARTED")

bio_medrxiv_papers = fetch_rxiv_both_servers(start_date, end_date, query_terms_list)
logging.info(f"FINISHED biomedrxiv, len: {len(bio_medrxiv_papers)}")
save_collected(bio_medrxiv_papers, 'data/0_bio_med.csv')



