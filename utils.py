import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import joblib

def fetch_document(url, tags=['p']):
    """
    Fetch the document from the given URL and extract text from specified HTML tags.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    article_text = ' '.join(paragraph.get_text() for paragraph in soup.find_all(tags))
    return article_text.strip()

def load_documents(path, tags=['p']):
    """
    Load documents from a CSV file and extract text from specified HTML tags.
    """
    documents = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            link = row[0]
            article_text = fetch_document(link, tags)
            if article_text:
                documents.append({"link": link, "text": article_text})
    
    return pd.DataFrame(documents)

def save_model(vectorizer, file_path):
    """
    Save the vectorizer model to the specified file.
    """
    joblib.dump(vectorizer, file_path)

def load_model(file_path):
    """
    Load the vectorizer model from the specified file.
    """
    return joblib.load(file_path)
