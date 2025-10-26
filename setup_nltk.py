"""
Setup script to download NLTK data for Streamlit Cloud
"""
import nltk
import ssl

def setup_nltk():
    """Download required NLTK data"""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required datasets
        print("Setting up NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK setup complete!")
        
    except Exception as e:
        print(f"NLTK setup failed: {e}")
        print("App will use fallback methods")

if __name__ == "__main__":
    setup_nltk()
