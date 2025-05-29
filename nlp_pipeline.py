import json
import random
import numpy as np
import re
import os
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

class IndonesianNLPPipeline:
    def __init__(self, model_type='svm'):
        # Indonesian text processing tools
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        # Model components
        self.vectorizer = TfidfVectorizer()
        self.model = None
        self.model_type = model_type
        
        # Data storage
        self.intents = None
        self.classes = None
        self.X = None
        self.y = None

    def load_intents(self, filepath):
        """Load intents from JSON file with validation"""
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if 'intents' not in data:
            raise ValueError("JSON file must contain 'intents' key")
            
        self.intents = data['intents']
        self._validate_intents()
        
    def _validate_intents(self):
        """Ensure minimum quality of training data"""
        for intent in self.intents:
            if len(intent['patterns']) < 5:
                print(f"Warning: Intent '{intent['tag']}' has only {len(intent['patterns'])} examples (recommend at least 5)")
            if not intent['responses']:
                raise ValueError(f"Intent '{intent['tag']}' has no responses")

    def preprocess_text(self, text):
        """Enhanced Indonesian text preprocessing"""
        # Case folding
        text = text.lower()
        
        # Remove special characters (keeping Indonesian letters)
        text = re.sub(r'[^\w\sà-üÀ-Ü]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Stemming
        text = self.stemmer.stem(text)
        
        # Stopword removal
        text = self.stopword_remover.remove(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def prepare_training_data(self):
        """Prepare training data"""
        documents = []
        classes = []
        
        for intent in self.intents:
            for pattern in intent['patterns']:
                processed = self.preprocess_text(pattern)
                documents.append(processed)
                classes.append(intent['tag'])
        
        self.classes = list(set(classes))
        self.y = classes
        self.X = self.vectorizer.fit_transform(documents)

    def train(self, test_size=0.2, random_state=42):
        """Enhanced training with cross-validation"""
        # Proper check for sparse or dense matrix
        if self.X is None or (hasattr(self.X, 'shape') and self.X.shape[0] == 0):
            raise ValueError("No training data available. Call prepare_training_data() first.")
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )
        
        # Initialize model with class weights
        if self.model_type == 'svm':
            self.model = SVC(
                kernel='linear', 
                probability=True, 
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced'
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Cross-validation
        print("\n=== Cross-validation ===")
        skf = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=skf, scoring='accuracy'
        )
        print(f"Fold Accuracies: {cv_scores}")
        print(f"Mean Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
        
        # Final training
        self.model.fit(X_train, y_train)
        
        # Evaluation
        print("\n=== Test Set Evaluation ===")
        y_pred = self.model.predict(X_test)
        print(classification_report(
            y_test, y_pred, 
            zero_division=0,
            target_names=self.classes
        ))

    def predict(self, text, confidence_threshold=0.65):
        """Improved prediction with confidence thresholding"""
        processed = self.preprocess_text(text)
        vector = self.vectorizer.transform([processed])
        
        if self.model:
            probas = self.model.predict_proba(vector)[0]
            best_idx = np.argmax(probas)
            confidence = probas[best_idx]
            intent = self.model.classes_[best_idx]
        else:
            # Fallback to similarity if no model
            return {
                'intent': None,
                'confidence': 0,
                'response': "Model not trained yet"
            }
        
        # Handle low confidence
        if confidence < confidence_threshold:
            return {
                'intent': 'unknown',
                'confidence': float(confidence),
                'response': "Maaf, saya tidak yakin. Bisa diulang dengan cara lain?"
            }
        
        # Find matching intent
        for item in self.intents:
            if item['tag'] == intent:
                return {
                    'intent': intent,
                    'confidence': float(confidence),
                    'response': random.choice(item['responses'])
                }
        
        return {
            'intent': 'unknown',
            'confidence': 0,
            'response': "Maaf, saya tidak mengerti"
        }

    def save_model(self, directory='model'):
        """Save complete model package"""
        os.makedirs(directory, exist_ok=True)
        
        # Save vectorizer
        with open(f"{directory}/tfidf.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save sklearn model
        if self.model:
            with open(f"{directory}/{self.model_type}_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'classes': self.classes,
            'intents': self.intents
        }
        with open(f"{directory}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_model(cls, directory='model'):
        """Load trained model package"""
        with open(f"{directory}/metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        pipeline = cls(model_type=metadata['model_type'])
        pipeline.intents = metadata['intents']
        pipeline.classes = metadata['classes']
        
        # Load vectorizer
        with open(f"{directory}/tfidf.pkl", 'rb') as f:
            pipeline.vectorizer = pickle.load(f)
        
        # Load model
        model_path = f"{directory}/{metadata['model_type']}_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                pipeline.model = pickle.load(f)
        
        return pipeline


def main():
    # Example usage
    nlp = IndonesianNLPPipeline(model_type='svm')
    
    try:
        # 1. Load intents
        nlp.load_intents('intents.json')
        
        # 2. Prepare data
        nlp.prepare_training_data()
        
        # 3. Train model
        nlp.train()
        
        # 4. Save model
        nlp.save_model()
        
        # 5. Test predictions
        test_phrases = [
            "Hai apa kabar?",
            "Tolong matikan lampu",
            "Tolong bukain atap karena hujan",
            "Apa kamu bisa berbahasa Indonesia?",
            "Masbro, tolong dong matiin lampu silau nih",
            "tolong matiin lampu dong",
            "ai, sebentar lagi hujan nih",
            "gelap nih",
            "terang banget nih",
            "hujan telah berhenti",
            "sekarang sudah menunjukan tanda tanda hujan"

        ]
        
        print("\n=== Predictions ===")
        for phrase in test_phrases:
            result = nlp.predict(phrase)
            print(f"\nInput: {phrase}")
            print(f"Intent: {result['intent']} ({result['confidence']:.2f})")
            print(f"Response: {result['response']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check:")
        print("- Your intents.json file exists and is valid JSON")
        print("- You have at least 5 examples per intent")
        print("- All intents have responses")

if __name__ == "__main__":
    main()