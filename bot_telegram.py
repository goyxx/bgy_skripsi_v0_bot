import os
import tempfile
import telebot
import requests
import speech_recognition as sr
from pydub import AudioSegment
from nlp_pipeline import IndonesianNLPPipeline  # Changed import

class BlynkController:
    def __init__(self, auth_token):
        self.base_url = "https://blynk.cloud/external/api"
        self.auth_token = auth_token
    
    def send_command(self, pin, value):
        """Send command to Blynk IoT platform"""
        url = f"{self.base_url}/update?token={self.auth_token}&{pin}={value}"
        
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Blynk API Error: {str(e)}")
            return False

# Configuration
BOT_TOKEN = "8129855163:AAE67IBAUxZSNWeVqqkQO4plTHkqi3HbS48" 
BLYNK_AUTH_TOKEN = "2v-Js6dAtUaNShPbbX0-BEKSacMyW-mu"  # Replace with your actual token
BLYNK_LAMPU_PIN = "V2"
BLYNK_ATAP_PIN = "V1"

# Initialize components
bot = telebot.TeleBot(BOT_TOKEN)
nlp = IndonesianNLPPipeline.load_model('model')  # Load trained model
blynk = BlynkController(BLYNK_AUTH_TOKEN)

def convert_voice_to_text(voice_message):
    """Convert voice message to text using Google Speech Recognition"""
    try:
        # Download voice file
        file_info = bot.get_file(voice_message.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp_ogg:
            tmp_ogg.write(downloaded_file)
            tmp_ogg_path = tmp_ogg.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            # Convert OGG to WAV
            AudioSegment.from_ogg(tmp_ogg_path).export(tmp_wav.name, format="wav")
            
            # Speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp_wav.name) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language='id-ID')
                return text
            
    except sr.UnknownValueError:
        raise Exception("Tidak dapat memahami audio")
    except sr.RequestError as e:
        raise Exception(f"Error pada layanan speech recognition: {str(e)}")
    except Exception as e:
        raise Exception(f"Error konversi suara: {str(e)}")
    finally:
        # Clean up temporary files
        if 'tmp_ogg_path' in locals() and os.path.exists(tmp_ogg_path):
            os.remove(tmp_ogg_path)
        if 'tmp_wav' in locals() and os.path.exists(tmp_wav.name):
            os.remove(tmp_wav.name)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Welcome message handler"""
    welcome_text = (
        "🤖 Bot Kontrol Rumah IoT 🤖\n\n"
        "Perintah yang tersedia:\n"
        "- Nyalakan/matikan lampu\n"
        "- Buka/tutup atap\n"
        "Anda bisa menggunakan perintah suara atau teks!"
    )
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def check_status(message):
    """Check device status"""
    try:
        lamp_status = "Menyala" if blynk.get_status(BLYNK_LAMPU_PIN) == "1" else "Mati"
        atap_status = "Terbuka" if blynk.get_status(BLYNK_ATAP_PIN) == "1" else "Tertutup"
        
        status_msg = (
            "🔄 Status Perangkat:\n"
            f"💡 Lampu: {lamp_status}\n"
            f"🏠 Atap: {atap_status}"
        )
        bot.reply_to(message, status_msg)
    except Exception as e:
        bot.reply_to(message, f"❌ Gagal memeriksa status: {str(e)}")

@bot.message_handler(content_types=['text', 'voice'])
def handle_message(message):
    """Main message handler for text and voice commands"""
    try:
        if message.content_type == 'voice':
            try:
                text = convert_voice_to_text(message.voice)
                bot.reply_to(message, f"🔊 Anda berkata: {text}")
            except Exception as e:
                bot.reply_to(message, f"❌ Error suara: {str(e)}")
                return
        else:
            text = message.text
        
        # Process with NLP
        result = nlp.predict(text)
        
        # Send response
        bot.reply_to(message, result['response'])
        
        # Execute command if confidence is high enough and intent is recognized
        if result['confidence'] > 0.65 and result['intent'] not in ['unknown', None]:
            success = False
            action = ""
            
            if result['intent'] == 'nyala_lampu':
                success = blynk.send_command(BLYNK_LAMPU_PIN, 1)
                action = "menyalakan lampu"
            elif result['intent'] == 'matikan_lampu':
                success = blynk.send_command(BLYNK_LAMPU_PIN, 0)
                action = "mematikan lampu"
            elif result['intent'] == 'buka_atap':
                success = blynk.send_command(BLYNK_ATAP_PIN, 1)
                action = "membuka atap"
            elif result['intent'] == 'tutup_atap':
                success = blynk.send_command(BLYNK_ATAP_PIN, 0)
                action = "menutup atap"
            else:
                return  # No action for other intents
            
            if success:
                bot.send_message(message.chat.id, f"✅ Berhasil {action}")
            else:
                bot.send_message(message.chat.id, f"❌ Gagal {action}")
    
    except Exception as e:
        bot.reply_to(message, f"❌ Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    print("🤖 Bot sedang berjalan...")
    bot.polling()
