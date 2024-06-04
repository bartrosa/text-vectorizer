import argparse
import json
from facade import TextVectorizationFacade

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
    facade = TextVectorizationFacade()
    
    if args.command == "train":
        vectorizer_params = json.loads(args.vectorizer_params)
        facade.train(args.train_file, args.vectorizer, vectorizer_params, args.model_file)
    elif args.command == "pick_best":
        facade.pick_best(args.test_file, args.queries_file, args.model_file, args.distance)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
