# text-vectorizer

## About

## Setup
```
sudo apt install python3.12-venv
sudo apt install python3-pip
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```
## Usage

```
python main.py train --train_file data/train.csv --vectorizer tfidf --vectorizer_params "{}" --model_file outputs/vectorizer.pkl
```

```
python main.py pick_best --test_file data/test.csv --queries_file data/queries.csv --model_file vectorizer.pkl --distance cosine
```

## Features

### Link parser