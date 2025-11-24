"""
Script to prepare Kaggle submission file from test data.
Loads test.csv, generates predictions using the trained model, and outputs
a submission file in the Kaggle format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from inference import load_classifier
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress function if tqdm is not available
    class tqdm:
        def __init__(self, iterable, desc=""):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable) if hasattr(iterable, '__len__') else None
            self.current = 0
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.current < len(self.iterable):
                item = self.iterable[self.current]
                self.current += 1
                if self.total and self.current % max(1, self.total // 20) == 0:
                    print(f"  {self.desc}: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)")
                return item
            raise StopIteration


def prepare_submission(test_csv_path, output_path, models_dir='models', batch_size=1000):
    """
    Prepare Kaggle submission file from test data.
    
    Args:
        test_csv_path: Path to test.csv file
        output_path: Path to save submission.csv
        models_dir: Directory containing saved model files
        batch_size: Number of comments to process in each batch
    """
    print("=" * 60)
    print("Kaggle Submission Preparation")
    print("=" * 60)
    
    # Load test data
    print(f"\n[1/4] Loading test data from {test_csv_path}...")
    try:
        test_df = pd.read_csv(test_csv_path)
        print(f"✓ Loaded {len(test_df):,} test comments")
        print(f"  Columns: {test_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"✗ Error: Test file not found at {test_csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading test data: {str(e)}")
        sys.exit(1)
    
    # Validate test data structure
    if 'id' not in test_df.columns:
        print("✗ Error: Test data must contain 'id' column")
        sys.exit(1)
    if 'comment_text' not in test_df.columns:
        print("✗ Error: Test data must contain 'comment_text' column")
        sys.exit(1)
    
    # Load classifier
    print(f"\n[2/4] Loading trained model from {models_dir}...")
    try:
        classifier = load_classifier(models_dir)
        print(f"✓ Model loaded: {classifier.metadata['model_name']} ({classifier.metadata['feature_type']})")
        print(f"  Target labels: {', '.join(classifier.target_cols)}")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        sys.exit(1)
    
    # Prepare submission data
    print(f"\n[3/4] Generating predictions for {len(test_df):,} comments...")
    print(f"  Processing in batches of {batch_size}...")
    
    # Initialize submission DataFrame
    submission_data = {
        'id': test_df['id'].values
    }
    
    # Initialize probability columns
    for col in classifier.target_cols:
        submission_data[col] = []
    
    # Process comments in batches with progress bar
    total_batches = (len(test_df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        
        # Process each comment in the batch
        for idx, row in batch_df.iterrows():
            comment_text = row['comment_text']
            
            # Handle missing/NaN values
            if pd.isna(comment_text) or comment_text is None:
                comment_text = ""
            
            # Get predictions
            try:
                results = classifier.predict(comment_text, threshold=0.5)
                
                # Extract probabilities for each label
                for col in classifier.target_cols:
                    prob = results[col]['probability']
                    submission_data[col].append(float(prob))
                    
            except Exception as e:
                # If prediction fails, set all probabilities to 0.0
                print(f"\nWarning: Prediction failed for comment {row['id']}: {str(e)}")
                for col in classifier.target_cols:
                    submission_data[col].append(0.0)
    
    # Create submission DataFrame
    print(f"\n[4/4] Creating submission file...")
    submission_df = pd.DataFrame(submission_data)
    
    # Ensure column order matches Kaggle format: id, toxic, severe_toxic, obscene, threat, insult, identity_hate
    column_order = ['id'] + classifier.target_cols
    submission_df = submission_df[column_order]
    
    # Validate submission format
    expected_cols = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    if list(submission_df.columns) != expected_cols:
        print(f"✗ Error: Column order mismatch!")
        print(f"  Expected: {expected_cols}")
        print(f"  Got: {list(submission_df.columns)}")
        sys.exit(1)
    
    # Validate probability ranges
    prob_cols = classifier.target_cols
    for col in prob_cols:
        min_prob = submission_df[col].min()
        max_prob = submission_df[col].max()
        if min_prob < 0.0 or max_prob > 1.0:
            print(f"⚠ Warning: {col} has values outside [0.0, 1.0]: [{min_prob:.4f}, {max_prob:.4f}]")
    
    # Save submission file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Submission file saved to: {output_path}")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Columns: {list(submission_df.columns)}")
    
    # Display summary statistics
    print(f"\n" + "=" * 60)
    print("Submission Summary")
    print("=" * 60)
    print(f"Total comments: {len(submission_df):,}")
    print(f"\nProbability Statistics:")
    for col in prob_cols:
        mean_prob = submission_df[col].mean()
        std_prob = submission_df[col].std()
        print(f"  {col:15s}: mean={mean_prob:.4f}, std={std_prob:.4f}")
    
    print(f"\n✓ Submission file ready for Kaggle!")
    print("=" * 60)


if __name__ == "__main__":
    # Default paths
    test_csv_path = "dataset/test/test.csv"
    output_path = "Kaggle_submission/submission.csv"
    models_dir = "models"
    
    # Allow command-line arguments
    if len(sys.argv) > 1:
        test_csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        models_dir = sys.argv[3]
    
    prepare_submission(test_csv_path, output_path, models_dir)

