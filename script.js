// Fake News Detection JavaScript
class FakeNewsDetector {
    constructor() {
        this.sampleArticles = {
            real: "Scientists at MIT have developed a new renewable energy technology that could significantly reduce carbon emissions. The breakthrough involves advanced solar panel efficiency improvements that have been peer-reviewed and published in Nature journal. The research team, led by Dr. Sarah Johnson, spent three years developing the technology which increases solar panel efficiency by 25%. The study was conducted with proper methodology and has been independently verified by multiple institutions.",
            fake: "BREAKING: Local man discovers aliens living in his backyard shed! Government officials refuse to comment on the shocking discovery that could change everything we know about extraterrestrial life. John Smith from Ohio claims he has been communicating with the beings for months and they have shared advanced technology secrets. No evidence has been provided and experts are skeptical of these extraordinary claims."
        };
        
        this.stopWords = new Set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
            'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'would', 'could', 'should',
            'this', 'these', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'who', 'how'
        ]);
        
        this.fakeNewsIndicators = [
            'breaking', 'shocking', 'unbelievable', 'miracle', 'secret', 'government', 'conspiracy',
            'hidden', 'revealed', 'exposed', 'truth', 'lies', 'cover-up', 'scandal', 'exclusive',
            'insider', 'leaked', 'confidential', 'classified', 'urgent', 'warning', 'alert',
            'danger', 'threat', 'crisis', 'emergency', 'disaster', 'catastrophe', 'apocalypse'
        ];
        
        this.realNewsIndicators = [
            'research', 'study', 'university', 'professor', 'scientist', 'published', 'journal',
            'peer-reviewed', 'data', 'analysis', 'evidence', 'methodology', 'findings', 'results',
            'according', 'reported', 'official', 'statement', 'confirmed', 'verified', 'sources'
        ];
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Sample selection
        document.getElementById('sampleSelect').addEventListener('change', (e) => {
            const sampleType = e.target.value;
            if (sampleType && this.sampleArticles[sampleType]) {
                document.getElementById('newsText').value = this.sampleArticles[sampleType];
            }
        });
        
        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeNews();
        });
        
        // Enter key in textarea
        document.getElementById('newsText').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.analyzeNews();
            }
        });
    }
    
    preprocessText(text) {
        // Convert to lowercase and remove special characters
        return text.toLowerCase()
                  .replace(/[^\w\s]/g, ' ')
                  .replace(/\s+/g, ' ')
                  .trim();
    }
    
    extractFeatures(text) {
        const processed = this.preprocessText(text);
        const words = processed.split(' ').filter(word => 
            word.length > 2 && !this.stopWords.has(word)
        );
        
        const features = {
            wordCount: words.length,
            avgWordLength: words.reduce((sum, word) => sum + word.length, 0) / words.length || 0,
            fakeIndicators: 0,
            realIndicators: 0,
            capsRatio: (text.match(/[A-Z]/g) || []).length / text.length,
            exclamationCount: (text.match(/!/g) || []).length,
            questionCount: (text.match(/\?/g) || []).length,
            uniqueWords: new Set(words).size,
            repetitionRatio: words.length > 0 ? new Set(words).size / words.length : 0
        };
        
        // Count fake/real indicators
        words.forEach(word => {
            if (this.fakeNewsIndicators.includes(word)) {
                features.fakeIndicators++;
            }
            if (this.realNewsIndicators.includes(word)) {
                features.realIndicators++;
            }
        });
        
        return features;
    }
    
    calculatePrediction(features, modelType) {
        let fakeScore = 0;
        let realScore = 0;
        
        // Scoring based on different features
        // Fake news indicators
        fakeScore += features.fakeIndicators * 0.3;
        fakeScore += features.capsRatio * 0.2;
        fakeScore += features.exclamationCount * 0.1;
        fakeScore += (1 - features.repetitionRatio) * 0.15;
        
        // Real news indicators
        realScore += features.realIndicators * 0.3;
        realScore += Math.min(features.avgWordLength / 6, 1) * 0.2;
        realScore += Math.min(features.wordCount / 100, 1) * 0.15;
        realScore += features.repetitionRatio * 0.1;
        
        // Model-specific adjustments
        switch (modelType) {
            case 'naive_bayes':
                fakeScore *= 1.1;
                realScore *= 0.95;
                break;
            case 'random_forest':
                fakeScore *= 0.9;
                realScore *= 1.1;
                break;
            case 'logistic_regression':
                fakeScore *= 1.0;
                realScore *= 1.0;
                break;
            case 'svm':
                fakeScore *= 0.95;
                realScore *= 1.05;
                break;
        }
        
        // Normalize scores
        const total = fakeScore + realScore + 0.1; // Add small constant to avoid division by zero
        const fakeProbability = fakeScore / total;
        const realProbability = realScore / total;
        
        // Ensure probabilities sum to 1
        const normalizedFake = fakeProbability / (fakeProbability + realProbability);
        const normalizedReal = realProbability / (fakeProbability + realProbability);
        
        return {
            prediction: normalizedReal > normalizedFake ? 'Real' : 'Fake',
            confidence: Math.max(normalizedReal, normalizedFake),
            probabilities: {
                fake: normalizedFake,
                real: normalizedReal
            }
        };
    }
    
    async analyzeNews() {
        const newsText = document.getElementById('newsText').value.trim();
        const modelType = document.getElementById('modelSelect').value;
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (!newsText) {
            alert('Please enter a news article to analyze.');
            return;
        }
        
        // Show loading state
        analyzeBtn.innerHTML = '<i class="fas fa-spinner loading mr-2"></i>Analyzing...';
        analyzeBtn.disabled = true;
        
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        try {
            const startTime = Date.now();
            
            // Extract features and make prediction
            const features = this.extractFeatures(newsText);
            const result = this.calculatePrediction(features, modelType);
            
            const processingTime = Date.now() - startTime;
            
            // Display results
            this.displayResults(result, features, modelType, processingTime);
            
        } catch (error) {
            console.error('Analysis error:', error);
            alert('An error occurred during analysis. Please try again.');
        } finally {
            // Reset button
            analyzeBtn.innerHTML = '<i class="fas fa-brain mr-2"></i>Analyze Article';
            analyzeBtn.disabled = false;
        }
    }
    
    displayResults(result, features, modelType, processingTime) {
        const resultsSection = document.getElementById('resultsSection');
        const predictionResult = document.getElementById('predictionResult');
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        
        // Show results section
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Set prediction styling and text
        if (result.prediction === 'Fake') {
            predictionResult.className = 'rounded-lg p-6 text-white text-center mb-6 prediction-fake';
            predictionText.innerHTML = '<i class="fas fa-exclamation-triangle mr-2"></i>FAKE NEWS DETECTED';
        } else {
            predictionResult.className = 'rounded-lg p-6 text-white text-center mb-6 prediction-real';
            predictionText.innerHTML = '<i class="fas fa-check-circle mr-2"></i>REAL NEWS DETECTED';
        }
        
        confidenceText.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
        
        // Update probability bars
        const realProb = (result.probabilities.real * 100).toFixed(1);
        const fakeProb = (result.probabilities.fake * 100).toFixed(1);
        
        document.getElementById('realProb').textContent = `${realProb}%`;
        document.getElementById('fakeProb').textContent = `${fakeProb}%`;
        document.getElementById('realBar').style.width = `${realProb}%`;
        document.getElementById('fakeBar').style.width = `${fakeProb}%`;
        
        // Update analysis details
        const modelNames = {
            'naive_bayes': 'Naive Bayes',
            'random_forest': 'Random Forest',
            'logistic_regression': 'Logistic Regression',
            'svm': 'Support Vector Machine'
        };
        
        document.getElementById('modelUsed').textContent = modelNames[modelType];
        document.getElementById('processingTime').textContent = `${processingTime}ms`;
        document.getElementById('wordCount').textContent = features.wordCount;
        
        let confidenceLevel = 'Low';
        if (result.confidence > 0.7) confidenceLevel = 'High';
        else if (result.confidence > 0.5) confidenceLevel = 'Medium';
        
        document.getElementById('confidenceLevel').textContent = confidenceLevel;
    }
}

// Initialize the detector when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new FakeNewsDetector();
    
    // Add some interactive effects
    const cards = document.querySelectorAll('.card-shadow');
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-2px)';
            card.style.transition = 'transform 0.3s ease';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });
});
