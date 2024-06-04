import csv
import requests
from bs4 import BeautifulSoup


def preprocess_text(path):
    print("Parser parsing...")
    # Open the CSV file and read the links
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            link = row[0]
            try:
                # Send a request to the link and get the HTML response
                response = requests.get(link, timeout=10)
                response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {link}: {e}")
                continue
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract the article text from the HTML content
            article_text = ''
            # TODO: Rethink usage of differents tags like ['p', 'ul', 'li', 'h1', 'h2', 'h3']
            for paragraph in soup.find_all(['p']):
                article_text += paragraph.get_text() + '\n\n'
            # Print the extracted article text
            print(article_text)
