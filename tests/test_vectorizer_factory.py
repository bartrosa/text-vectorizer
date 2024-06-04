import pytest
from src.vectorizer_factory import VectorizerFactory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

def test_create_vectorizer():
    factory = VectorizerFactory()
    
    vectorizer_params = {'max_features': 1000}
    
    count_vectorizer = factory.create_vectorizer('count', vectorizer_params)
    assert isinstance(count_vectorizer, CountVectorizer)
    assert count_vectorizer.max_features == 1000
    
    tfidf_vectorizer = factory.create_vectorizer('tfidf', vectorizer_params)
    assert isinstance(tfidf_vectorizer, TfidfVectorizer)
    assert tfidf_vectorizer.max_features == 1000
    
    hashing_vectorizer = factory.create_vectorizer('hashing', {'n_features': 1000})
    assert isinstance(hashing_vectorizer, HashingVectorizer)
    assert hashing_vectorizer.n_features == 1000

    with pytest.raises(ValueError):
        factory.create_vectorizer('unknown', vectorizer_params)

if __name__ == '__main__':
    pytest.main()
