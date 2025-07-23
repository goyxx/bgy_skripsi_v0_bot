import json
import random
import numpy as np
import re
import os
import pickle

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Scikit-learn imports (tetap sama)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report


# --- Tambahan untuk NLTK: Unduh resource yang dibutuhkan jika belum ada ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')
try:
    stopwords.words('indonesian')
except LookupError:
    print("Downloading NLTK 'stopwords' for Indonesian...")
    nltk.download('stopwords')
# --- Akhir tambahan NLTK ---

class IndonesianNLPPipelineNLTK:
    ### MODIFIKASI DIMULAI ###
    # Tambahkan intents_filepath ke constructor untuk menyimpan path ke file intents.json
    def __init__(self, model_type='svm', intents_filepath='intents.json'):
    ### MODIFIKASI SELESAI ###
        # NLTK for text processing
        self.stop_words_indonesian = set(stopwords.words('indonesian'))
        
        # Model components
        self.vectorizer = TfidfVectorizer()
        self.model = None
        self.model_type = model_type
        
        # Data storage
        self.intents = None
        self.classes = None
        self.X = None
        self.y = None
        ### MODIFIKASI DIMULAI ###
        self.intents_filepath = intents_filepath # Simpan path ke file intents
        ### MODIFIKASI SELESAI ###


    def load_intents(self, filepath):
        """Load intents from JSON file with validation"""
        ### MODIFIKASI DIMULAI ###
        self.intents_filepath = filepath # Simpan path setiap kali file dimuat
        ### MODIFIKASI SELESAI ###
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
        """Enhanced Indonesian text preprocessing using NLTK"""
        # Case folding
        text = text.lower()
        
        # Remove special characters (keeping Indonesian letters)
        text = re.sub(r'[^\w\sà-üÀ-Ü]', '', text) # Mempertahankan karakter umlaut/aksen jika ada di B.Indonesia
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenization (NLTK)
        words = word_tokenize(text)
        
        # Stopword removal (NLTK)
        filtered_words = [word for word in words if word not in self.stop_words_indonesian]
        
        # Re-join words
        text = " ".join(filtered_words)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def prepare_training_data(self):
        """Prepare training data"""
        documents = []
        classes = []
        
        if not self.intents:
            raise ValueError("Intents not loaded. Call load_intents() first.")

        for intent in self.intents:
            for pattern in intent['patterns']:
                processed = self.preprocess_text(pattern)
                if processed: # Hanya tambahkan jika hasil praproses tidak kosong
                    documents.append(processed)
                    classes.append(intent['tag'])
        
        if not documents:
            raise ValueError("No documents were generated after preprocessing. Check your patterns and preprocessing steps.")

        self.classes = sorted(list(set(classes))) # Urutkan agar konsisten
        self.y = classes
        self.X = self.vectorizer.fit_transform(documents)

    def train(self, test_size=0.2, random_state=42):
        """Enhanced training with cross-validation"""
        if self.X is None or (hasattr(self.X, 'shape') and self.X.shape[0] == 0):
            raise ValueError("No training data available. Call prepare_training_data() first.")
        
        if not self.y:
                raise ValueError("No labels (y) available for training. Call prepare_training_data() first.")

        # Pastikan y memiliki cukup anggota untuk stratifikasi
        unique_classes_in_y = np.unique(self.y)
        min_samples_per_class = np.min(np.bincount(np.array([self.classes.index(label) for label in self.y])))
        
        # n_splits untuk StratifiedKFold tidak boleh lebih besar dari jumlah sampel di kelas terkecil
        n_splits_cv = min(5, min_samples_per_class if min_samples_per_class > 1 else 2)

        if n_splits_cv < 2 : # Tidak bisa melakukan cross-validation atau train-test split
            print("Warning: Not enough samples or classes for stratified split or cross-validation. Training on whole data.")
            X_train, X_test, y_train, y_test = self.X, self.X, self.y, self.y # Latih dan uji pada data yang sama
            if self.X.shape[0] == 0:
                    raise ValueError("Training data X is empty, cannot proceed.")
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=self.y
                )
            except ValueError as e:
                print(f"Could not perform stratified split: {e}. Falling back to non-stratified split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, 
                    test_size=test_size, 
                    random_state=random_state
                )

        if X_train.shape[0] == 0:
            raise ValueError("X_train is empty after split. Not enough data for training.")
            
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
                class_weight='balanced',
                random_state=random_state
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB() # MultinomialNB tidak punya class_weight='balanced' secara langsung
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Cross-validation
        print("\n=== Cross-validation ===")
        if n_splits_cv >= 2 and X_train.shape[0] >= n_splits_cv :
            skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
            try:
                cv_scores = cross_val_score(
                    self.model, X_train, y_train, 
                    cv=skf, scoring='accuracy'
                )
                print(f"Fold Accuracies: {cv_scores}")
                print(f"Mean Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
            except ValueError as e:
                    print(f"Could not perform cross-validation: {e}. It might be due to too few samples in a class.")
        else:
            print("Skipping cross-validation due to insufficient samples or splits.")

        # Final training
        self.model.fit(X_train, y_train)
        
        # Evaluation
        if X_test.shape[0] > 0:
            print("\n=== Test Set Evaluation ===")
            y_pred = self.model.predict(X_test)
            print(classification_report(
                y_test, y_pred, 
                zero_division=0,
                target_names=self.classes # Pastikan ini menggunakan self.classes yang sudah diurutkan
            ))
        else:
            print("\n=== Test Set Evaluation ===")
            print("Skipping test set evaluation as X_test is empty.")

    ### MODIFIKASI DIMULAI ###
    # Tambahkan parameter auto_enrich_log_file untuk mencatat kalimat baru
    def predict(self, text, confidence_threshold=0.65, auto_enrich_log_file=None):
    ### MODIFIKASI SELESAI ###
        """
        Improved prediction with confidence thresholding and low-confidence feedback.
        """
        if not self.model:
            return {
                'intent': None,
                'confidence': 0,
                'response': "Model not trained yet or training failed."
            }
        if not self.vectorizer:
            return {
                'intent': None,
                'confidence': 0,
                'response': "Vectorizer not fitted yet."
            }

        processed = self.preprocess_text(text)
        if not processed: # Jika hasil praproses kosong
            return {
                'intent': 'unknown',
                'confidence': 0,
                'response': "Maaf, input Anda terlalu pendek atau tidak mengandung kata yang dikenali setelah diproses."
            }
            
        vector = self.vectorizer.transform([processed])
        
        probas = self.model.predict_proba(vector)[0]
        best_idx = np.argmax(probas)
        confidence = probas[best_idx]
        predicted_intent_tag = self.model.classes_[best_idx] 
        
        if confidence >= confidence_threshold:
            # Jika cukup yakin, cari respons dan jalankan tindakan
            for item in self.intents:
                if item['tag'] == predicted_intent_tag:
                    return {
                        'intent': predicted_intent_tag,
                        'confidence': float(confidence),
                        'response': random.choice(item['responses'])
                    }
            # Fallback jika tag tidak ditemukan (seharusnya tidak terjadi)
            return {
                'intent': 'unknown_tag_mismatch',
                'confidence': float(confidence),
                'response': "Maaf, terjadi kesalahan dalam mencocokkan respons."
            }
        else:
            ### MODIFIKASI DIMULAI ###
            # Jika TIDAK cukup yakin, catat kalimat ke file log jika pathnya diberikan
            if auto_enrich_log_file:
                # Siapkan data yang akan dicatat: kalimat asli dan tebakan tag
                log_entry = {
                    "raw_pattern": text,
                    "predicted_tag": predicted_intent_tag,
                    "confidence": float(confidence)
                }
                # Buka file dalam mode 'append' (a) untuk menambahkan di akhir file
                try:
                    with open(auto_enrich_log_file, 'a', encoding='utf-8') as f:
                        # Tulis sebagai satu baris JSON, diikuti newline
                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    print(f"LOGGED: Kalimat '{text}' dicatat untuk tag '{predicted_intent_tag}'")
                except Exception as e:
                    print(f"ERROR LOGGING: Gagal menulis ke file log {auto_enrich_log_file}. Error: {e}")
            ### MODIFIKASI SELESAI ###

            # Beri tahu pengguna tag mana yang paling mendekati.
            return {
                'intent': f"low_confidence_guess ({predicted_intent_tag})",
                'confidence': float(confidence),
                'response': f"Saya rasa ini tentang '{predicted_intent_tag}', tapi saya belum cukup yakin. Bisa perjelas maksud Anda?"
            }

    def save_model(self, directory='model_nltk'): # Ubah nama direktori default jika mau
        """Save complete model package"""
        os.makedirs(directory, exist_ok=True)
        
        with open(f"{directory}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        if self.model:
            with open(f"{directory}/{self.model_type}_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
        
        metadata = {
            'model_type': self.model_type,
            'classes': self.classes, # Simpan classes yang terurut
            'intents': self.intents # Sebaiknya intents juga disimpan untuk referensi
        }
        with open(f"{directory}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Model saved to {directory}/")

    @classmethod
    def load_model(cls, directory='model_nltk'): # Ubah nama direktori default jika mau
        """Load trained model package"""
        metadata_path = f"{directory}/metadata.json"
        vectorizer_path = f"{directory}/tfidf_vectorizer.pkl"

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        ### MODIFIKASI DIMULAI ###
        # Saat memuat model, kita tidak tahu path intents.json aslinya,
        # jadi kita akan set path ini secara manual di bot setelah memuat.
        pipeline = cls(model_type=metadata['model_type'])
        ### MODIFIKASI SELESAI ###
        pipeline.intents = metadata['intents']
        pipeline.classes = metadata['classes'] # Muat classes yang terurut
        
        with open(vectorizer_path, 'rb') as f:
            pipeline.vectorizer = pickle.load(f)
        
        model_path = f"{directory}/{metadata['model_type']}_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                pipeline.model = pickle.load(f)
        else:
            print(f"Warning: Model file {model_path} not found. Model will be None.")
            pipeline.model = None # Eksplisit set ke None
            
        print(f"Model loaded from {directory}/")
        return pipeline

    ### FUNGSI BARU DIMULAI ###
    def learn_from_log(self, log_filepath, model_save_dir='model_indonesian_nltk'):
        """
        Membaca file log, memperbarui intents.json, melatih ulang, dan menyimpan model baru.
        """
        print("\n=== MEMULAI PROSES PEMBELAJARAN OTOMATIS ===")
        
        # 1. Pastikan file log ada dan tidak kosong
        if not os.path.exists(log_filepath) or os.path.getsize(log_filepath) == 0:
            print("Tidak ada data baru untuk dipelajari. Proses dihentikan.")
            return False

        # 2. Muat ulang intents dari file JSON utama (self.intents_filepath) untuk data terbaru
        self.load_intents(self.intents_filepath)

        # 3. Baca log dan tambahkan pattern baru
        new_patterns_added = 0
        all_logs = []
        with open(log_filepath, 'r', encoding='utf-8') as f:
            all_logs = f.readlines()

        for line in all_logs:
            try:
                log_entry = json.loads(line)
                new_pattern = log_entry['raw_pattern']
                target_tag = log_entry['predicted_tag']
                
                # Cari intent yang sesuai dalam data yang sudah dimuat
                for intent in self.intents:
                    if intent['tag'] == target_tag:
                        # Cek duplikasi (case-insensitive) sebelum menambahkan
                        if new_pattern.lower() not in [p.lower() for p in intent['patterns']]:
                            intent['patterns'].append(new_pattern)
                            new_patterns_added += 1
                            print(f"  -> Menambahkan pattern baru '{new_pattern}' ke tag '{target_tag}'")
                        break # Lanjut ke log berikutnya setelah menemukan tag
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  -> Melewati baris log yang tidak valid: {line.strip()}. Error: {e}")

        if new_patterns_added == 0:
            print("Tidak ada pattern unik yang baru untuk ditambahkan.")
            # Hapus file log agar tidak diproses lagi
            os.remove(log_filepath)
            print(f"File log '{log_filepath}' telah dibersihkan.")
            return False

        # 4. Simpan intents.json yang sudah diperbarui
        try:
            with open(self.intents_filepath, 'w', encoding='utf-8') as f:
                json.dump({'intents': self.intents}, f, ensure_ascii=False, indent=4)
            print(f"\nBerhasil memperbarui {self.intents_filepath} dengan {new_patterns_added} pattern baru.")
        except Exception as e:
            print(f"FATAL: Gagal menyimpan intents.json. Error: {e}")
            return False


        # 5. Latih ulang model dengan data yang sudah diperkaya
        print("\nMelatih ulang model dengan dataset baru...")
        self.prepare_training_data()
        self.train()

        # 6. Simpan model yang sudah dilatih ulang
        self.save_model(directory=model_save_dir)
        print(f"\nModel baru berhasil dilatih dan disimpan di '{model_save_dir}'.")

        # 7. Hapus file log setelah selesai diproses
        os.remove(log_filepath)
        print(f"File log '{log_filepath}' telah dibersihkan.")
        print("=== PROSES PEMBELAJARAN SELESAI ===")
        return True
    ### FUNGSI BARU SELESAI ###

# Kode di bawah ini untuk testing, tidak akan diubah
def main():
    # Ganti nama kelas ke versi NLTK
    nlp = IndonesianNLPPipelineNLTK(model_type='svm') 
    
    # Buat file intents.json sederhana jika belum ada
    intents_file = 'intents.json'
    if not os.path.exists(intents_file):
        print("INTENTS TAK DITEMUKAN")
        sample_intents = {
            "intents": [
                {
                    "tag": "sapaan",
                    "patterns": ["halo", "hai", "hei", "selamat pagi", "selamat siang", "apa kabar"],
                    "responses": ["Halo!", "Hai juga!", "Ada yang bisa dibantu?", "Kabar baik, bagaimana denganmu?"]
                },
                {
                    "tag": "lampu_nyala",
                    "patterns": ["nyalakan lampu", "hidupkan lampu", "gelap nih tolong lampu", "lampunya mati", "tolong nyalain lampu"],
                    "responses": ["Baik, lampu akan dinyalakan.", "Siap, lampu menyala sekarang."]
                },
                {
                    "tag": "lampu_mati",
                    "patterns": ["matikan lampu", "padamkan lampu", "terang banget lampunya", "lampu tolong dimatikan", "tolong matiin lampu"],
                    "responses": ["Oke, lampu akan dimatikan.", "Lampu sudah dipadamkan."]
                },
                {
                    "tag": "info_hujan",
                    "patterns": ["apakah akan hujan", "cuaca hari ini gimana", "bakal hujan gak", "prediksi hujan", "sepertinya mau hujan"],
                    "responses": ["Saya belum bisa cek info cuaca saat ini.", "Sebaiknya cek aplikasi cuaca untuk info akurat."]
                }
            ]
        }
        with open(intents_file, 'w', encoding='utf-8') as f:
            json.dump(sample_intents, f, indent=2, ensure_ascii=False)
        print(f"'{intents_file}' created with sample data.")

    try:
        # 1. Load intents
        nlp.load_intents(intents_file)
        
        # 2. Prepare data
        nlp.prepare_training_data()
        
        # 3. Train model
        nlp.train()
        
        # 4. Save model
        nlp.save_model(directory='model_indonesian_nltk') # Simpan ke direktori berbeda
        
        # 5. Test predictions
        test_phrases = [
            "Hai apa kabar?",
            "Tolong matikan lampu",
            # "Tolong bukain atap karena hujan", # Mungkin tidak ada intentnya
            "Apa kamu bisa berbahasa Indonesia?", # Mungkin tidak ada intentnya
            "Masbro, tolong dong matiin lampu silau nih",
            "tolong matiin lampu dong",
            "ai, sebentar lagi hujan nih",
            "gelap nih", # Bisa ambigu, bisa lampu_nyala
            "terang banget nih", # Bisa ambigu, bisa lampu_mati
            "matiin sunroofnya dong",
            "matikan sunroofnya dong",
            "Nyalain Dong",
            "Atap buka",
            "Tutup Atap",
            "Ko agak gelap ya",
            "cuaca hari ini gelap sekali seperti mau hujan",
            "gerah sekali disini",
            "digin sekali disini",
            "panas banget tolong",
            "sangat gelap, tolong nyalakan lampu"

            # "hujan telah berhenti", # Mungkin tidak ada intentnya
            # "sekarang sudah menunjukan tanda tanda hujan" # Mungkin tidak ada intentnya
        ]
        
        print("\n=== Predictions (New Model) ===")
        for phrase in test_phrases:
            result = nlp.predict(phrase)
            print(f"\nInput: {phrase}")
            print(f"Intent: {result.get('intent', 'N/A')} ({result.get('confidence', 0):.2f})")
            print(f"Response: {result.get('response', 'N/A')}")

        print("\n=== Loading and Testing Loaded Model ===")
        loaded_nlp = IndonesianNLPPipelineNLTK.load_model(directory='model_indonesian_nltk')
        
        for phrase in test_phrases:
            result = loaded_nlp.predict(phrase)
            print(f"\nInput (loaded): {phrase}")
            print(f"Intent: {result.get('intent', 'N/A')} ({result.get('confidence', 0):.2f})")
            print(f"Response: {result.get('response', 'N/A')}")

    
    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        print("Pastikan data Anda cukup dan formatnya benar.")
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: {str(fnfe)}")
        print("Pastikan file intents.json ada di direktori yang sama atau path-nya benar.")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {str(e)}")
        print(traceback.format_exc())
        print("Please check:")
        print("- Your intents.json file exists and is valid JSON")
        print("- You have enough examples per intent (minimum 2-3 for stratified splitting to work, recommend at least 5)")
        print("- All intents have responses")
        print("- NLTK resources ('punkt', 'stopwords') are downloaded.")

if __name__ == "__main__":
    main()