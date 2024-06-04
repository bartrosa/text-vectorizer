import joblib

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
