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

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_theme_css(theme):
    """Get CSS styles based on selected theme"""
    if theme == 'dark':
        return """
        <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            .main-header {
                font-size: 3rem;
                color: #64b5f6;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 0 0 10px rgba(100, 181, 246, 0.3);
            }
            .prediction-box {
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }
            .fake-news {
                background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%);
                border-left: 5px solid #ff1744;
                color: white;
            }
            .real-news {
                background: linear-gradient(135deg, #388e3c 0%, #4caf50 100%);
                border-left: 5px solid #00e676;
                color: white;
            }
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 999;
                background: #262730;
                border: 2px solid #64b5f6;
                border-radius: 25px;
                padding: 8px 16px;
                color: #64b5f6;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .theme-toggle:hover {
                background: #64b5f6;
                color: #262730;
                transform: scale(1.05);
            }
            .sidebar .sidebar-content {
                background-color: #262730;
            }
            .metric-container {
                background-color: #262730;
                padding: 1rem;
                border-radius: 10px;
                border: 1px solid #404040;
            }
            .stSelectbox > div > div {
                background-color: #262730;
                color: #fafafa;
            }
            .stTextArea > div > div > textarea {
                background-color: #262730;
                color: #fafafa;
                border: 1px solid #404040;
            }
        </style>
        """
    else:  # light theme
        return """
        <style>
            .stApp {
                background-color: #ffffff;
                color: #262626;
            }
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 0 2px 4px rgba(31, 119, 180, 0.1);
            }
            .prediction-box {
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }
            .fake-news {
                background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
                border-left: 5px solid #f44336;
                color: #d32f2f;
            }
            .real-news {
                background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
                border-left: 5px solid #4caf50;
                color: #2e7d32;
            }
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 999;
                background: #f5f5f5;
                border: 2px solid #1f77b4;
                border-radius: 25px;
                padding: 8px 16px;
                color: #1f77b4;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .theme-toggle:hover {
                background: #1f77b4;
                color: white;
                transform: scale(1.05);
            }
            .metric-container {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                border: 1px solid #e9ecef;
            }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    detector = FakeNewsDetector()
    detector.load_data()
    detector.prepare_features()
    detector.train_models()
    return detector

def main():
    # Theme toggle in sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Theme selector
    theme_options = {
        "🌞 Light Mode": "light",
        "🌙 Dark Mode": "dark"
    }
    
    current_theme_label = "🌞 Light Mode" if st.session_state.theme == "light" else "🌙 Dark Mode"
    selected_theme = st.sidebar.selectbox(
        "Choose Theme:",
        options=list(theme_options.keys()),
        index=list(theme_options.values()).index(st.session_state.theme),
        key="theme_selector"
    )
    
    # Update theme if changed
    new_theme = theme_options[selected_theme]
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
    
    # Add theme toggle button in the top right
    theme_icon = "🌙" if st.session_state.theme == "light" else "🌞"
    theme_text = "Dark" if st.session_state.theme == "light" else "Light"
    
    st.markdown(f"""
    <div class="theme-toggle" onclick="toggleTheme()">
        {theme_icon} {theme_text} Mode
    </div>
    <script>
    function toggleTheme() {{
        // This will be handled by the sidebar selector
        console.log('Theme toggle clicked');
    }}
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.markdown("---")
    st.sidebar.title("📋 Navigation")
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
                    
                    # Confidence meter with themed styling
                    st.subheader("📊 Confidence Analysis")
                    
                    # Create themed metric containers
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #f44336; margin: 0;">🚨 Fake News Probability</h4>
                            <h2 style="margin: 10px 0;">{result['probabilities']['Fake']:.1%}</h2>
                            <div style="background: #ffcdd2; height: 10px; border-radius: 5px; overflow: hidden;">
                                <div style="background: #f44336; height: 100%; width: {result['probabilities']['Fake']*100}%; transition: width 0.5s ease;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #4caf50; margin: 0;">✅ Real News Probability</h4>
                            <h2 style="margin: 10px 0;">{result['probabilities']['Real']:.1%}</h2>
                            <div style="background: #c8e6c9; height: 10px; border-radius: 5px; overflow: hidden;">
                                <div style="background: #4caf50; height: 100%; width: {result['probabilities']['Real']*100}%; transition: width 0.5s ease;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Overall confidence indicator
                    st.markdown("### 🎯 Overall Confidence")
                    confidence_color = "#4caf50" if result['confidence'] > 0.7 else "#ff9800" if result['confidence'] > 0.5 else "#f44336"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <div style="display: inline-block; padding: 10px 20px; background: {confidence_color}; color: white; border-radius: 25px; font-size: 1.2em; font-weight: bold;">
                            {result['confidence']:.1%} Confident
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
    
    # Performance chart with theme-aware styling
    st.subheader("📈 Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Theme-aware colors
    if st.session_state.theme == 'dark':
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        text_color = '#fafafa'
        grid_color = '#404040'
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        text_color = '#262626'
        grid_color = '#e0e0e0'
    
    # Create gradient bars
    colors = ['#64b5f6', '#ff7043', '#66bb6a', '#ab47bc']
    bars = ax.bar(performance_df['Model'], performance_df['Accuracy'], 
                  color=colors, alpha=0.8, edgecolor=text_color, linewidth=1.5)
    
    # Styling
    ax.set_ylabel('Accuracy', color=text_color, fontsize=12)
    ax.set_title('Model Performance Comparison', color=text_color, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.tick_params(colors=text_color)
    ax.grid(True, alpha=0.3, color=grid_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
    - **🎨 Theme Support**: Beautiful dark and light modes for better user experience
    - **📱 Responsive Design**: Works seamlessly on all devices
    
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
    
    # Theme showcase
    st.markdown("---")
    st.markdown("## 🎨 Theme Showcase")
    
    current_theme = st.session_state.theme
    theme_demo_col1, theme_demo_col2 = st.columns(2)
    
    with theme_demo_col1:
        st.markdown(f"""
        <div class="metric-container" style="text-align: center;">
            <h4>🌞 Light Mode</h4>
            <p>Clean, bright interface perfect for daytime use</p>
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%); padding: 20px; border-radius: 10px; border: 2px solid {'#1f77b4' if current_theme == 'light' else '#ccc'};">
                <div style="color: #1f77b4; font-weight: bold;">Currently {'Active' if current_theme == 'light' else 'Inactive'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with theme_demo_col2:
        st.markdown(f"""
        <div class="metric-container" style="text-align: center;">
            <h4>🌙 Dark Mode</h4>
            <p>Easy on the eyes for extended use and low-light environments</p>
            <div style="background: linear-gradient(135deg, #0e1117 0%, #262730 100%); padding: 20px; border-radius: 10px; border: 2px solid {'#64b5f6' if current_theme == 'dark' else '#ccc'};">
                <div style="color: #64b5f6; font-weight: bold;">Currently {'Active' if current_theme == 'dark' else 'Inactive'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("💡 **Tip**: Switch themes using the selector in the sidebar to see the difference!")
    
    st.markdown("---")
    st.markdown("**Developed for educational and research purposes**")
    
    # Add current theme indicator
    theme_icon = "🌙" if current_theme == "dark" else "🌞"
    st.markdown(f"<div style='text-align: center; opacity: 0.7; margin-top: 20px;'>Currently using {theme_icon} {current_theme.title()} Mode</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
