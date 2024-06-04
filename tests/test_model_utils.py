import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from src.model_utils import save_model, load_model

def test_save_and_load_model(tmp_path):
    vectorizer = TfidfVectorizer()
    model_file = tmp_path / 'vectorizer.pkl'

    save_model(vectorizer, model_file)
    assert os.path.exists(model_file)

    loaded_vectorizer = load_model(model_file)
    assert isinstance(loaded_vectorizer, TfidfVectorizer)

if __name__ == '__main__':
    pytest.main()
