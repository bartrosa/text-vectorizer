# text-vectorizer
## Task
Create a program that can train a model for text vectorization and select the most relevant text from a database based on a query.

## Detailed requirements:

- The program should be run from the command line.
- The program should have separate commands for training (train) and for selecting the most relevant text (pick_best).
- The solution should include a user manual with necessary steps to run the program.
- Three files are provided:
    - Each file contains links to documents, each link on a new line.
    - train.csv - contains links to documents to be used for train (training set).
    - test.csv - contains links to documents among which the most relevant text should be selected.
    - queries.csv - contains links to documents that are queries.

- Training Procedure (train):

    - The training set is provided as input to the program.
    - Each text in the training set should undergo preprocessing:
        - Only the main part of the article should be processed (excluding headers, navigation, or footer - constant elements of the page).
        - Each text should be cleaned of HTML tags (so that the text remains human-readable).
    - The preprocessed texts should be passed to the .fit method of a Vectorizer object. The program should support 3 classes:
        - CountVectorizer
        - HashingVectorizer
        - TfidfVectorizer
    - The result of the training procedure is a trained Vectorizer, which should be saved to a file.
    The type of Vectorizer object is provided as input to the program.
    - Parameters for the Vectorizer instance:
        - Minimum: use default or choose your own.
        - Optimal: parameters are provided as input to the program.

- Selecting the Most Relevant Match (pick_best):

    - Texts should undergo the same preprocessing as in the training procedure.
    - The previously trained model is provided as input to the program.
    - The set of documents to search through is provided as input to the program.
    - Minimum: a query, in the form of a clean URL, is provided as input to the program.
    - Optimal: the program additionally supports providing an entire set of queries as input from a file.
    - The most relevant document is considered to be the one whose vector is closest to the vector of the query text. - - The distance between vectors is calculated, with the smallest distance indicating the closest match.
    - The program should support the following distance measures: cosine, euclidean, manhattan.
    - The distance measure is provided as input to the program.
    - Program output:
        - Minimum: URL of the best-matched document (if a single URL query is provided as input).
        - Optimal: A list of links if an entire set of queries is provided as input.


## Setup
Commands to setup environment.
```
sudo apt install python3.12-venv
sudo apt install python3-pip
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```
## Usage
### Tool run
Training run:
```
python run.py train --train_file data/train.csv --vectorizer tfidf --vectorizer_params "{}" --model_file outputs/vectorizer.pkl
```
- ```--vectorizer_type``` Type of vectorizer (count for CountVectorizer, hasging for HashingVectorizer, tfidf for TfidfVectorizer)
- ``--input`` Path to the training CSV file
- ``--output`` Path to save the trained model
params: Additional parameters for the vectorizer (optional)

Picking best fit:
```
python run.py pick_best --test_file data/test.csv --queries_file data/queries.csv --model_file outputs/vectorizer.pkl --distance cosine
```
- ``--model`` Path to the trained model file
- ``--input`` Path to the CSV file containing documents to search
- ``--query`` URL of the query document
distance: Distance metric (cosine, euclidean, manhattan)
### Unit tests run

```
pytest tests/
```
