import streamlit as st
import pandas as pd
import numpy as np
import os

# Setup NLTK first
try:
    from setup_nltk import setup_nltk
    setup_nltk()
except Exception as e:
    st.warning(f"NLTK setup warning: {e}")

from fake_news_detector import FakeNewsDetector
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fake-news {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .real-news {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    detector = FakeNewsDetector()
    detector.load_data()
    detector.prepare_features()
    detector.train_models()
    return detector

def main():
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Analysis", "About"])
    
    if page == "Prediction":
        prediction_page()
    elif page == "Model Analysis":
        analysis_page()
    else:
        about_page()

def prediction_page():
    st.header("🔍 News Article Prediction")
    
    # Load model
    with st.spinner("Loading model..."):
        detector = load_model()
    
    # Input section
    st.subheader("Enter News Article")
    
    # Sample articles for quick testing
    sample_articles = {
        "Select a sample...": "",
        "Real News Sample": "Scientists at MIT have developed a new renewable energy technology that could significantly reduce carbon emissions. The breakthrough involves advanced solar panel efficiency improvements that have been peer-reviewed and published in Nature journal.",
        "Fake News Sample": "BREAKING: Local man discovers aliens living in his backyard shed! Government officials refuse to comment on the shocking discovery that could change everything we know about extraterrestrial life."
    }
    
    selected_sample = st.selectbox("Quick test with sample articles:", list(sample_articles.keys()))
    
    if selected_sample != "Select a sample...":
        news_text = st.text_area("News Article Text:", value=sample_articles[selected_sample], height=150)
    else:
        news_text = st.text_area("News Article Text:", height=150, placeholder="Paste your news article here...")
    
    # Model selection
    model_options = ['Naive Bayes', 'Random Forest', 'Logistic Regression', 'SVM']
    selected_model = st.selectbox("Select Model:", model_options)
    
    # Prediction
    if st.button("🔍 Analyze Article", type="primary"):
        if news_text.strip():
            with st.spinner("Analyzing article..."):
                try:
                    result = detector.predict_news(news_text, selected_model)
                    
                    # Display result
                    if result['prediction'] == 'Fake':
                        st.markdown(f"""
                        <div class="prediction-box fake-news">
                            <h3>🚨 FAKE NEWS DETECTED</h3>
                            <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                            <p><strong>Fake Probability:</strong> {result['probabilities']['Fake']:.2%}</p>
                            <p><strong>Real Probability:</strong> {result['probabilities']['Real']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box real-news">
                            <h3>✅ REAL NEWS DETECTED</h3>
                            <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                            <p><strong>Fake Probability:</strong> {result['probabilities']['Fake']:.2%}</p>
                            <p><strong>Real Probability:</strong> {result['probabilities']['Real']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.subheader("Confidence Meter")
                    confidence_col1, confidence_col2 = st.columns(2)
                    
                    with confidence_col1:
                        st.metric("Fake News Probability", f"{result['probabilities']['Fake']:.2%}")
                    with confidence_col2:
                        st.metric("Real News Probability", f"{result['probabilities']['Real']:.2%}")
                    
                    # Progress bars
                    st.progress(result['probabilities']['Fake'])
                    st.caption("Fake News Probability")
                    
                except Exception as e:
                    st.error(f"Error analyzing article: {str(e)}")
        else:
            st.warning("Please enter a news article to analyze.")

def analysis_page():
    st.header("📊 Model Analysis & Performance")
    
    # Load model
    with st.spinner("Loading model and generating analysis..."):
        detector = load_model()
        results = detector.evaluate_models()
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    # Create performance dataframe
    performance_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    performance_df = performance_df.sort_values('Accuracy', ascending=False)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    for i, (model, accuracy) in enumerate(performance_df.values):
        if i == 0:
            col1.metric(model, f"{accuracy:.2%}", "Best Model")
        elif i == 1:
            col2.metric(model, f"{accuracy:.2%}")
        elif i == 2:
            col3.metric(model, f"{accuracy:.2%}")
        else:
            col4.metric(model, f"{accuracy:.2%}")
    
    # Performance chart
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(performance_df['Model'], performance_df['Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Dataset information
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", len(detector.df))
    with col2:
        st.metric("Real News", sum(detector.df['label']))
    with col3:
        st.metric("Fake News", len(detector.df) - sum(detector.df['label']))
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in detector.trained_models:
        st.subheader("Feature Importance (Random Forest)")
        rf_model = detector.trained_models['Random Forest']
        feature_names = detector.vectorizer.get_feature_names_out()
        importances = rf_model.feature_importances_
        
        # Get top 20 features
        top_indices = np.argsort(importances)[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = [importances[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_features, top_importances)
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Most Important Features')
        plt.tight_layout()
        st.pyplot(fig)

def about_page():
    st.header("ℹ️ About Fake News Detection System")
    
    st.markdown("""
    ## Overview
    This Fake News Detection System uses machine learning to classify news articles as either **Real** or **Fake**. 
    The system employs multiple algorithms to provide robust predictions.
    
    ## Features
    - **Multiple ML Models**: Naive Bayes, Random Forest, Logistic Regression, and SVM
    - **Text Preprocessing**: Advanced NLP techniques including stemming and stopword removal
    - **TF-IDF Vectorization**: Converts text to numerical features
    - **Interactive Interface**: Easy-to-use web interface for real-time predictions
    - **Model Comparison**: Performance analysis and visualization
    
    ## How It Works
    1. **Text Preprocessing**: Clean and normalize the input text
    2. **Feature Extraction**: Convert text to TF-IDF vectors
    3. **Model Prediction**: Use trained ML models to classify the article
    4. **Confidence Score**: Provide probability scores for both classes
    
    ## Models Used
    - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
    - **Random Forest**: Ensemble method using multiple decision trees
    - **Logistic Regression**: Linear model for binary classification
    - **SVM**: Support Vector Machine for complex pattern recognition
    
    ## Dataset
    The current version uses a sample dataset for demonstration. For production use, 
    consider training on larger datasets like:
    - LIAR dataset
    - FakeNewsNet
    - ISOT Fake News Dataset
    
    ## Technical Stack
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning library
    - **NLTK**: Natural language processing
    - **Streamlit**: Web interface framework
    - **Pandas/NumPy**: Data manipulation
    - **Matplotlib/Seaborn**: Data visualization
    
    ## Usage Tips
    - Longer articles generally provide better predictions
    - The system works best with news-style content
    - Consider the confidence score when interpreting results
    - Try different models to compare predictions
    
    ## Limitations
    - Performance depends on training data quality
    - May not generalize well to all types of misinformation
    - Context and real-world knowledge are not considered
    - Sample dataset is limited for demonstration purposes
    """)
    
    st.markdown("---")
    st.markdown("**Developed for educational and research purposes**")

if __name__ == "__main__":
    main()
