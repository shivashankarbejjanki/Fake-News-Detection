import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self):
        """Initialize the Fake News Detector"""
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.trained_models = {}
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        stop_words = set(stopwords.words('english'))
        tokens = [self.stemmer.stem(token) for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def load_data(self, file_path=None):
        """Load and prepare the dataset"""
        if file_path is None:
            # Create a sample dataset for demonstration
            print("No dataset provided. Creating sample data for demonstration...")
            sample_data = {
                'title': [
                    "Scientists discover new planet in solar system",
                    "BREAKING: Aliens land in New York City",
                    "Stock market reaches new high amid economic recovery",
                    "Miracle cure for cancer found in grandmother's kitchen",
                    "Government announces new education policy",
                    "Celebrity spotted with mysterious alien technology",
                    "Research shows benefits of regular exercise",
                    "Local man claims he can predict the future with 100% accuracy",
                    "New study reveals impact of climate change",
                    "SHOCKING: Time travel proven real by local scientist"
                ],
                'text': [
                    "Astronomers have confirmed the discovery of a new celestial body in our solar system using advanced telescopes.",
                    "Multiple witnesses report seeing alien spacecraft landing in Times Square yesterday evening.",
                    "The stock market closed at record highs today as investors remain optimistic about economic indicators.",
                    "A 90-year-old grandmother claims her secret recipe can cure any form of cancer within days.",
                    "The education ministry has unveiled comprehensive reforms to improve student outcomes nationwide.",
                    "Paparazzi photos show famous actor using what appears to be extraterrestrial communication device.",
                    "A comprehensive study involving 10,000 participants confirms the health benefits of daily physical activity.",
                    "John Smith from Ohio says he has never been wrong about future events and offers predictions for sale.",
                    "Climate scientists present new data showing accelerated changes in global weather patterns.",
                    "Local university professor claims to have successfully sent objects back in time using household items."
                ],
                'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
            }
            self.df = pd.DataFrame(sample_data)
        else:
            self.df = pd.read_csv(file_path)
        
        # Combine title and text for better feature extraction
        self.df['content'] = self.df['title'].fillna('') + ' ' + self.df['text'].fillna('')
        
        print(f"Dataset loaded successfully with {len(self.df)} articles")
        print(f"Real news: {sum(self.df['label'])}, Fake news: {len(self.df) - sum(self.df['label'])}")
        
        return self.df
    
    def prepare_features(self):
        """Preprocess text and create feature vectors"""
        print("Preprocessing text data...")
        
        # Preprocess the content
        self.df['processed_content'] = self.df['content'].apply(self.preprocess_text)
        
        # Create TF-IDF features
        X = self.vectorizer.fit_transform(self.df['processed_content'])
        y = self.df['label']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
    
    def train_models(self):
        """Train all models"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        results = {}
        
        plt.figure(figsize=(15, 10))
        
        for i, (name, model) in enumerate(self.trained_models.items(), 1):
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = accuracy
            
            # Confusion Matrix
            plt.subplot(2, 2, i)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} - Accuracy: {accuracy:.4f}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('c:/Users/shiva/OneDrive/Documents/Desktop/ML/Fake news Detection/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        for name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20}: {accuracy:.4f}")
        
        return results
    
    def predict_news(self, text, model_name='Naive Bayes'):
        """Predict if a news article is fake or real"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        model = self.trained_models[model_name]
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        result = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probability),
            'probabilities': {
                'Fake': probability[0],
                'Real': probability[1]
            }
        }
        
        return result
    
    def generate_word_clouds(self):
        """Generate word clouds for real and fake news"""
        fake_news = ' '.join(self.df[self.df['label'] == 0]['processed_content'])
        real_news = ' '.join(self.df[self.df['label'] == 1]['processed_content'])
        
        plt.figure(figsize=(15, 6))
        
        # Fake news word cloud
        plt.subplot(1, 2, 1)
        wordcloud_fake = WordCloud(width=400, height=300, background_color='white').generate(fake_news)
        plt.imshow(wordcloud_fake, interpolation='bilinear')
        plt.title('Fake News Word Cloud')
        plt.axis('off')
        
        # Real news word cloud
        plt.subplot(1, 2, 2)
        wordcloud_real = WordCloud(width=400, height=300, background_color='white').generate(real_news)
        plt.imshow(wordcloud_real, interpolation='bilinear')
        plt.title('Real News Word Cloud')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('c:/Users/shiva/OneDrive/Documents/Desktop/ML/Fake news Detection/word_clouds.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the fake news detection pipeline"""
    print("="*60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("="*60)
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load data (using sample data for demonstration)
    detector.load_data()
    
    # Prepare features
    detector.prepare_features()
    
    # Train models
    detector.train_models()
    
    # Evaluate models
    results = detector.evaluate_models()
    
    # Generate visualizations
    detector.generate_word_clouds()
    
    # Interactive prediction
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION")
    print("="*50)
    
    while True:
        news_text = input("\nEnter a news article to check (or 'quit' to exit): ")
        if news_text.lower() == 'quit':
            break
        
        try:
            result = detector.predict_news(news_text)
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities - Fake: {result['probabilities']['Fake']:.4f}, Real: {result['probabilities']['Real']:.4f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
