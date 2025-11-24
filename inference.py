"""
Inference module for toxic comment classification.
Provides functions to load the saved model and make predictions.
"""

import joblib
import json
import numpy as np
import pandas as pd
import re
import spacy
from pathlib import Path


class ToxicCommentClassifier:
    """Class for loading and using the toxic comment classification model."""
    
    def __init__(self, models_dir='models'):
        """
        Initialize the classifier by loading the saved model and metadata.
        
        Args:
            models_dir: Directory containing saved model files
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.target_cols = None
        self.feature_type = None
        self.nlp = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the saved model, vectorizer, and metadata."""
        # Load metadata
        metadata_path = self.models_dir / 'model_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.target_cols = self.metadata['target_columns']
        self.feature_type = self.metadata['feature_type']
        
        # Load model
        model_path = self.models_dir / self.metadata['model_file']
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"Loaded model: {self.metadata['model_name']} ({self.feature_type})")
        
        # Load vectorizer if using TF-IDF
        if self.feature_type == 'TF-IDF':
            vectorizer_path = self.models_dir / self.metadata['vectorizer_file']
            if not vectorizer_path.exists():
                raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            self.vectorizer = joblib.load(vectorizer_path)
            print("Loaded TF-IDF vectorizer")
        else:
            # Load spaCy model for embeddings
            try:
                self.nlp = spacy.load('en_core_web_md')
                print("Loaded spaCy model for word embeddings")
            except OSError:
                try:
                    self.nlp = spacy.load('en_core_web_lg')
                    print("Loaded spaCy model (large) for word embeddings")
                except OSError:
                    raise OSError("spaCy model not found. Please install: python -m spacy download en_core_web_md")
    
    def clean_text(self, text):
        """Clean text for feature extraction."""
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_comment_embedding(self, text):
        """Extract average word embedding for a comment."""
        doc = self.nlp(text)
        vectors = [token.vector for token in doc if token.has_vector]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.nlp.vocab.vectors.shape[1])
    
    def preprocess(self, text):
        """Preprocess text based on feature type."""
        cleaned_text = self.clean_text(text)
        
        if self.feature_type == 'TF-IDF':
            return self.vectorizer.transform([cleaned_text])
        else:
            embedding = self.get_comment_embedding(cleaned_text)
            return embedding.reshape(1, -1)
    
    def predict(self, text, threshold=0.5):
        """
        Predict toxicity labels for a given text.
        
        Args:
            text: Input text string
            threshold: Probability threshold for binary predictions (default: 0.5)
        
        Returns:
            Dictionary with probabilities and binary predictions for all 6 labels
        """
        # Preprocess text
        features = self.preprocess(text)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            # Handle OneVsRestClassifier output
            if isinstance(probabilities, list):
                probabilities = np.array([prob[:, 1] for prob in probabilities]).T
            else:
                probabilities = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
        else:
            # For models without predict_proba, use predict
            predictions = self.model.predict(features)
            probabilities = predictions[0] if len(predictions.shape) > 1 else predictions
        
        # Ensure probabilities is 1D array with 6 values
        if probabilities.ndim > 1:
            probabilities = probabilities[0]
        
        # Create results dictionary
        results = {}
        for i, col in enumerate(self.target_cols):
            prob = float(probabilities[i]) if i < len(probabilities) else 0.0
            results[col] = {
                'probability': prob,
                'prediction': 1 if prob >= threshold else 0
            }
        
        return results
    
    def predict_batch(self, texts, threshold=0.5):
        """
        Predict toxicity labels for multiple texts.
        
        Args:
            texts: List of input text strings
            threshold: Probability threshold for binary predictions (default: 0.5)
        
        Returns:
            List of dictionaries with predictions for each text
        """
        results = []
        for text in texts:
            results.append(self.predict(text, threshold))
        return results
    
    def format_results(self, results, include_probabilities=True):
        """
        Format prediction results for display.
        
        Args:
            results: Dictionary or list of dictionaries from predict/predict_batch
            include_probabilities: Whether to include probability scores
        
        Returns:
            Formatted results as DataFrame or dictionary
        """
        if isinstance(results, list):
            # Batch results
            data = []
            for i, result in enumerate(results):
                row = {}
                for col in self.target_cols:
                    if include_probabilities:
                        row[f'{col}_prob'] = result[col]['probability']
                    row[col] = result[col]['prediction']
                data.append(row)
            return pd.DataFrame(data)
        else:
            # Single result
            formatted = {}
            for col in self.target_cols:
                if include_probabilities:
                    formatted[f'{col}_probability'] = results[col]['probability']
                formatted[col] = results[col]['prediction']
            return formatted


def load_classifier(models_dir='models'):
    """
    Convenience function to load the classifier.
    
    Args:
        models_dir: Directory containing saved model files
    
    Returns:
        ToxicCommentClassifier instance
    """
    return ToxicCommentClassifier(models_dir)


if __name__ == "__main__":
    # Example usage
    classifier = load_classifier()
    
    # Test prediction
    test_text = "This is a test comment to check if the model works correctly."
    results = classifier.predict(test_text)
    print("\nPrediction Results:")
    print(json.dumps(results, indent=2))
    
    # Format results
    formatted = classifier.format_results(results)
    print("\nFormatted Results:")
    print(pd.DataFrame([formatted]))

