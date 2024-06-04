import argparse
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from utils import load_documents, save_model, load_model

def get_vectorizer(vectorizer_type, vectorizer_params):
    """
    Get the appropriate vectorizer based on the specified type and parameters.
    """
    if vectorizer_type == "count":
        return CountVectorizer(**vectorizer_params)
    elif vectorizer_type == "tfidf":
        return TfidfVectorizer(**vectorizer_params)
    elif vectorizer_type == "hashing":
        return HashingVectorizer(**vectorizer_params)
    else:
        raise ValueError("Invalid vectorizer type")

def train(args):
    """
    Train a vectorizer model with the given training data and save the model.
    """
    documents = load_documents(args.train_file)
    vectorizer_params = json.loads(args.vectorizer_params)
    
    vectorizer = get_vectorizer(args.vectorizer, vectorizer_params)
    text_only = documents['text'].tolist()
    vectorizer.fit(text_only)
    
    save_model(vectorizer, args.model_file)
    print(f"Model saved to {args.model_file}")

def pick_best(args):
    """
    Pick the best matching document for each query using the trained vectorizer model.
    """
    test_df = load_documents(args.test_file)
    query_df = load_documents(args.queries_file)
    
    test_documents = test_df['text'].tolist()
    query_documents = query_df['text'].tolist()
    
    vectorizer = load_model(args.model_file)
    doc_vectors = vectorizer.transform(test_documents)
    query_vectors = vectorizer.transform(query_documents)
    
    if args.distance == "cosine":
        distance_func = cosine_distances
    elif args.distance == "euclidean":
        distance_func = euclidean_distances
    elif args.distance == "manhattan":
        distance_func = manhattan_distances
    else:
        raise ValueError("Invalid distance type")
    
    results = []
    for query_vec in query_vectors:
        distances = distance_func(query_vec, doc_vectors)
        best_match_index = np.argmin(distances)
        results.append(test_df.iloc[best_match_index]["link"])
    
    print("\nBest matching links:")
    for result in results:
        print(result)

def main():
    parser = argparse.ArgumentParser(description="Text vectorization and matching")
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train_file", type=str, required=True)
    train_parser.add_argument("--vectorizer", type=str, choices=["count", "tfidf", "hashing"], required=True)
    train_parser.add_argument("--vectorizer_params", type=str, default="{}")
    train_parser.add_argument("--model_file", type=str, required=True)
    
    pick_best_parser = subparsers.add_parser("pick_best")
    pick_best_parser.add_argument("--test_file", type=str, required=True)
    pick_best_parser.add_argument("--queries_file", type=str, required=True)
    pick_best_parser.add_argument("--model_file", type=str, required=True)
    pick_best_parser.add_argument("--distance", type=str, choices=["cosine", "euclidean", "manhattan"], required=True)
    
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "pick_best":
        pick_best(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
