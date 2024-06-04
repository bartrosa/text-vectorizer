from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

class VectorizerFactory:
    """Factory for creating vectorizer instances."""

    def create_vectorizer(self, vectorizer_type, vectorizer_params):
        """
        Create a vectorizer based on the specified type and parameters.
        """
        if vectorizer_type == "count":
            return CountVectorizer(**vectorizer_params)
        elif vectorizer_type == "tfidf":
            return TfidfVectorizer(**vectorizer_params)
        elif vectorizer_type == "hashing":
            return HashingVectorizer(**vectorizer_params)
        else:
            raise ValueError("Invalid vectorizer type")
