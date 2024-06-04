from .vectorizer_factory import VectorizerFactory
from .document_loader import DocumentLoader
from .model_utils import save_model, load_model
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
import numpy as np

class TextVectorizationFacade:
    """Facade for text vectorization and matching operations."""

    def __init__(self):
        self.document_loader = DocumentLoader()
        self.vectorizer_factory = VectorizerFactory()

    def train(self, train_file, vectorizer_type, vectorizer_params, model_file):
        """
        Train a vectorizer model and save it to a file.
        """
        documents = self.document_loader.load_documents(train_file)
        vectorizer = self.vectorizer_factory.create_vectorizer(vectorizer_type, vectorizer_params)
        
        text_only = documents['text'].tolist()
        vectorizer.fit(text_only)
        
        save_model(vectorizer, model_file)
        print(f"Model saved to {model_file}")

    def pick_best(self, test_file, queries_file, model_file, distance_metric):
        """
        Pick the best matching documents for each query.
        """
        test_df = self.document_loader.load_documents(test_file)
        query_df = self.document_loader.load_documents(queries_file)
        
        test_documents = test_df['text'].tolist()
        query_documents = query_df['text'].tolist()
        
        vectorizer = load_model(model_file)
        doc_vectors = vectorizer.transform(test_documents)
        query_vectors = vectorizer.transform(query_documents)
        
        distance_func = self.get_distance_func(distance_metric)
        
        results = []
        for query_vec in query_vectors:
            distances = distance_func(query_vec, doc_vectors)
            best_match_index = np.argmin(distances)
            results.append(test_df.iloc[best_match_index]["link"])
        
        print("Best matching links:")
        for result in results:
            print(result)

    @staticmethod
    def get_distance_func(distance_metric):
        """
        Get the distance function based on the specified metric.
        """
        if distance_metric == "cosine":
            return cosine_distances
        elif distance_metric == "euclidean":
            return euclidean_distances
        elif distance_metric == "manhattan":
            return manhattan_distances
        else:
            raise ValueError("Invalid distance type")
