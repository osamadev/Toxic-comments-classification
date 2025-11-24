"""
Streamlit application for toxic comment classification.
Provides a user-friendly interface to classify comments for toxicity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from inference import load_classifier
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .toxic-label {
        color: #d62728;
        font-weight: bold;
    }
    .safe-label {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    try:
        with st.spinner("Loading model..."):
            st.session_state.classifier = load_classifier()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

classifier = st.session_state.classifier

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Threshold slider
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for binary predictions. Higher values require higher confidence."
    )
    
    st.markdown("---")
    
    # Model information
    st.subheader("📊 Model Information")
    st.write(f"**Algorithm:** {classifier.metadata['model_name']}")
    st.write(f"**Feature Type:** {classifier.metadata['feature_type']}")
    st.write(f"**Macro F1 Score:** {classifier.metadata['metrics']['macro_f1']:.4f}")
    st.write(f"**Micro F1 Score:** {classifier.metadata['metrics']['micro_f1']:.4f}")
    
    st.markdown("---")
    
    # Instructions
    st.subheader("ℹ️ Instructions")
    st.markdown("""
    1. Enter a comment in the text area
    2. Click 'Classify Comment' to get predictions
    3. View all 6 toxicity labels with probabilities
    4. Adjust threshold to change sensitivity
    """)

# Main content
st.markdown('<h1 class="main-header">⚠️ Toxic Comment Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Classify comments for multiple types of toxicity")

# Tabs for different input methods
tab1, tab2 = st.tabs(["📝 Single Comment", "📁 Batch Upload"])

with tab1:
    st.subheader("Enter a Comment")
    
    # Text input
    comment_text = st.text_area(
        "Comment Text",
        height=150,
        placeholder="Enter a comment to classify...",
        help="Type or paste a comment to analyze for toxicity"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        classify_button = st.button("🔍 Classify Comment", type="primary", use_container_width=True)
    
    if classify_button and comment_text.strip():
        with st.spinner("Analyzing comment..."):
            # Get predictions
            results = classifier.predict(comment_text, threshold=threshold)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Classification Results")
            
            # Create two columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Probability bars
                st.markdown("### Probability Scores")
                probabilities = [results[col]['probability'] for col in classifier.target_cols]
                predictions = [results[col]['prediction'] for col in classifier.target_cols]
                
                # Create DataFrame for visualization
                df_viz = pd.DataFrame({
                    'Label': [col.replace('_', ' ').title() for col in classifier.target_cols],
                    'Probability': probabilities,
                    'Prediction': ['Toxic' if p == 1 else 'Safe' for p in predictions]
                })
                
                # Create bar chart
                fig = px.bar(
                    df_viz,
                    x='Label',
                    y='Probability',
                    color='Prediction',
                    color_discrete_map={'Toxic': '#d62728', 'Safe': '#2ca02c'},
                    title="Toxicity Probabilities by Label",
                    labels={'Probability': 'Probability', 'Label': 'Toxicity Type'},
                    height=400
                )
                fig.update_layout(
                    yaxis_range=[0, 1],
                    xaxis_tickangle=-45,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary metrics
                st.markdown("### Summary")
                num_toxic = sum(predictions)
                st.metric("Toxic Labels Detected", f"{num_toxic}/6")
                
                max_prob_label = classifier.target_cols[np.argmax(probabilities)]
                max_prob_value = max(probabilities)
                st.metric("Highest Probability", f"{max_prob_value:.2%}", delta=max_prob_label.replace('_', ' ').title())
            
            # Detailed results table
            st.markdown("### Detailed Results")
            results_data = []
            for col in classifier.target_cols:
                results_data.append({
                    'Label': col.replace('_', ' ').title(),
                    'Probability': f"{results[col]['probability']:.4f}",
                    'Prediction': '✅ Toxic' if results[col]['prediction'] == 1 else '❌ Safe',
                    'Confidence': f"{results[col]['probability']*100:.1f}%"
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Label": st.column_config.TextColumn("Toxicity Type", width="medium"),
                    "Probability": st.column_config.NumberColumn("Probability", format="%.4f"),
                    "Prediction": st.column_config.TextColumn("Binary Prediction"),
                    "Confidence": st.column_config.TextColumn("Confidence %")
                }
            )
            
            # Show formatted results similar to training data
            st.markdown("### Results in Training Data Format")
            formatted_results = classifier.format_results(results, include_probabilities=True)
            st.json(formatted_results)
    
    elif classify_button and not comment_text.strip():
        st.warning("⚠️ Please enter a comment to classify.")

with tab2:
    st.subheader("Upload CSV File for Batch Classification")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with a 'comment_text' column"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            if 'comment_text' not in df.columns:
                st.error("❌ CSV file must contain a 'comment_text' column")
            else:
                st.success(f"✅ Loaded {len(df)} comments")
                
                # Show preview
                st.markdown("### File Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Process button
                if st.button("🚀 Process All Comments", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each comment
                    all_results = []
                    for idx, row in df.iterrows():
                        comment = str(row['comment_text'])
                        results = classifier.predict(comment, threshold=threshold)
                        all_results.append(results)
                        
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {idx + 1}/{len(df)} comments...")
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'comment_text': df['comment_text'].values,
                        **{col: [r[col]['prediction'] for r in all_results] for col in classifier.target_cols},
                        **{f'{col}_prob': [r[col]['probability'] for r in all_results] for col in classifier.target_cols}
                    })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Processed {len(df)} comments successfully!")
                    
                    # Display results
                    st.markdown("### Classification Results")
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="toxic_classification_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.markdown("### Summary Statistics")
                    summary_data = []
                    for col in classifier.target_cols:
                        toxic_count = results_df[col].sum()
                        summary_data.append({
                            'Label': col.replace('_', ' ').title(),
                            'Toxic Count': toxic_count,
                            'Percentage': f"{(toxic_count/len(df))*100:.2f}%",
                            'Avg Probability': f"{results_df[f'{col}_prob'].mean():.4f}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Toxic Comment Classification System | Multi-Label Classification"
    "</div>",
    unsafe_allow_html=True
)

