# Toxic Comments Classification System

A comprehensive machine learning system for classifying toxic comments with multi-label support. This project includes data analysis, model training, evaluation, and a user-friendly Streamlit web application for real-time inference.

## Features

- **Multi-Label Classification**: Classifies comments across 6 toxicity types:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate

- **Multiple ML Algorithms**: Compares Logistic Regression, Random Forest, and Multi-Layer Perceptron
- **Feature Extraction Methods**: Uses both TF-IDF and spaCy word embeddings
- **Comprehensive Evaluation**: Includes accuracy, F1-score, recall, precision, and AUC metrics
- **Interactive Web App**: Streamlit interface for easy comment classification

## Project Structure

```
toxic-comments-classification/
├── dataset/
│   ├── train/
│   │   └── train.csv
│   └── test/
│       └── test.csv
├── models/                          # Saved models (created after training)
│   ├── best_model_*.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json
├── toxic-comments-classification-analysis.ipynb  # Main analysis notebook
├── inference.py                      # Inference helper module
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### 1. Clone or Download the Project

```bash
cd Toxic-comments-classification
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_md
```

## Usage

### Training and Analysis

1. **Run the Analysis Notebook**:
   - Open `toxic-comments-classification-analysis.ipynb` in Jupyter
   - Execute all cells to:
     - Perform data analysis
     - Train multiple models
     - Compare performance
     - Save the best model

2. **Model Training**:
   - The notebook will automatically:
     - Train 3 algorithms (Logistic Regression, Random Forest, MLP)
     - Use 2 feature extraction methods (TF-IDF, Word Embeddings)
     - Evaluate all 6 combinations
     - Select and save the best performing model

### Running the Streamlit App

1. **Ensure Model is Saved**:
   - Run the notebook first to generate the model files in the `models/` directory

2. **Start the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Use the Application**:
   - The app will open in your default web browser
   - Enter a comment in the text area
   - Click "Classify Comment" to see predictions
   - View all 6 toxicity labels with probabilities
   - Adjust the threshold slider to change sensitivity

### Using the Inference Module

You can also use the inference module programmatically:

```python
from inference import load_classifier

# Load the classifier
classifier = load_classifier()

# Predict a single comment
results = classifier.predict("Your comment here", threshold=0.5)

# Print results
for label, data in results.items():
    print(f"{label}: {data['prediction']} (probability: {data['probability']:.4f})")

# Predict multiple comments
comments = ["Comment 1", "Comment 2", "Comment 3"]
batch_results = classifier.predict_batch(comments, threshold=0.5)

# Format results as DataFrame
formatted = classifier.format_results(batch_results)
print(formatted)
```

## Model Information

The system automatically selects the best model based on Macro F1-score. The saved model includes:

- **Model file**: Trained classifier (`.pkl` format)
- **Vectorizer** (if using TF-IDF): TF-IDF vectorizer for feature extraction
- **Metadata**: JSON file containing:
  - Model performance metrics
  - Per-class metrics
  - Target column names
  - Feature extraction method
  - Timestamp

## Understanding Multi-Label Predictions

The system provides predictions for all 6 toxicity labels simultaneously:

- **Probability Scores**: Range from 0.0 to 1.0 for each label
- **Binary Predictions**: 0 (Safe) or 1 (Toxic) based on threshold (default: 0.5)
- **Multi-Label Support**: A comment can have multiple toxic labels at once

### Example Output Format

```json
{
  "toxic": {"probability": 0.85, "prediction": 1},
  "severe_toxic": {"probability": 0.12, "prediction": 0},
  "obscene": {"probability": 0.78, "prediction": 1},
  "threat": {"probability": 0.05, "prediction": 0},
  "insult": {"probability": 0.92, "prediction": 1},
  "identity_hate": {"probability": 0.08, "prediction": 0}
}
```

## Streamlit App Features

### Single Comment Classification
- Text input area for entering comments
- Real-time classification with all 6 labels
- Interactive probability bar charts
- Detailed results table
- JSON output matching training data format

### Batch Processing
- Upload CSV files with `comment_text` column
- Process multiple comments at once
- Download results as CSV
- Summary statistics for all labels

### Customization
- Adjustable prediction threshold (0.0 - 1.0)
- Model information display
- Performance metrics sidebar

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Troubleshooting

### Model Not Found Error
- Ensure you've run the notebook and saved the model
- Check that `models/` directory exists with required files

### spaCy Model Error
- Run: `python -m spacy download en_core_web_md`
- If medium model unavailable, the system will try large or small models

### Streamlit Not Starting
- Verify installation: `pip install streamlit`
- Check port availability (default: 8501)

## Performance Metrics

The system evaluates models using:
- **Macro F1-Score**: Average F1 across all classes
- **Micro F1-Score**: Overall F1 considering all samples
- **Hamming Loss**: Multi-label classification error
- **AUC-ROC**: Area under ROC curve per class
- **Per-Class Metrics**: Accuracy, Precision, Recall, F1 for each label

## Future Improvements

- Fine-tune the hyperparameters of the models using GridSearch and cross validation 
- Ensemble methods combining multiple models
- Advanced handling for rare classes (SMOTE, focal loss)
- Transformer-based models (BERT, DistilBERT)

