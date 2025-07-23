import os
import tempfile
import telebot
import requests
import speech_recognition as sr
from pydub import AudioSegment
from nlp_pipeline import IndonesianNLPPipelineNLTK
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

class BlynkController:
    def __init__(self, auth_token):
        self.base_url = "https://blynk.cloud/external/api"
        self.auth_token = auth_token

    def send_command(self, pin, value):
        url = f"{self.base_url}/update?token={self.auth_token}&{pin}={value}"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Blynk API Error sending command: {str(e)}")
            return False

    def get_status(self, pin):
        url = f"{self.base_url}/get?token={self.auth_token}&{pin}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.text.strip() 
            else:
                print(f"Blynk API Error getting status: HTTP {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Blynk API Error getting status: {str(e)}")
            return None

# --- Configuration ---
BOT_TOKEN = "8129855163:AAE67IBAUxZSNWeVqqkQO4plTHkqi3HbS48"
BLYNK_AUTH_TOKEN = "fJC9H-1iW7ZVfimMlgDdu37KaGAecN1c"

# Pin untuk Aktor
BLYNK_LAMPU_PIN = "V2"
BLYNK_ATAP_PIN = "V1"
BLYNK_KIPAS_PIN = "V3"

# Pin untuk Sensor
BLYNK_RAINDROP_PIN = "V4"
BLYNK_LDR_PIN = "V5"
BLYNK_DHT_TEMP_PIN = "V0" # Mengirim nilai suhu saja

# Pin Kontrol Mode
BLYNK_MODE_PIN = "V10"

# --- Nilai Ambang Batas Sensor (Thresholds) ---
TEMP_THRESHOLD_HOT = 30.0
LDR_THRESHOLD_DARK = 2000
RAIN_THRESHOLD_WET = 1800

# --- Konfigurasi lain ---
INTENTS_JSON_PATH = 'intents.json'
MODEL_DIR = 'model_indonesian_nltk'
AUTO_ENRICH_LOG_FILE = 'new_patterns_to_learn.jsonl'
ADMIN_USER_ID = 6180905254

# --- Inisialisasi ---
bot = telebot.TeleBot(BOT_TOKEN)
nlp = IndonesianNLPPipelineNLTK.load_model(MODEL_DIR)
nlp.intents_filepath = INTENTS_JSON_PATH
blynk = BlynkController(BLYNK_AUTH_TOKEN)

current_mode = "manual" # Default mode

# --- Fungsi-fungsi ---
def convert_voice_to_text(voice_message):
    """Convert voice message to text using Google Speech Recognition"""
    # Mendefinisikan path sementara di awal agar bisa diakses di blok finally
    tmp_ogg_path = None
    tmp_wav_path = None
    
    try:
        file_info = bot.get_file(voice_message.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Membuat file ogg sementara
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp_ogg:
            tmp_ogg.write(downloaded_file)
            tmp_ogg_path = tmp_ogg.name

        # Membuat file wav sementara
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
            # Konversi dari ogg ke wav
            AudioSegment.from_ogg(tmp_ogg_path).export(tmp_wav_path, format="wav")

        # Proses pengenalan suara
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav_path) as source:
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
        # Membersihkan file sementara yang telah dibuat
        if tmp_ogg_path and os.path.exists(tmp_ogg_path):
            os.remove(tmp_ogg_path)
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

# --- HANDLER UNTUK PERINTAH ---
@bot.message_handler(commands=['start', 'help', 'mode', 'status', 'learn'])
def handle_commands(message):
    global nlp
    command = message.text.split()[0][1:]

    if command in ['start', 'help']:
        welcome_text = (
            "🤖 *Bot Kontrol Rumah IoT* 🤖\n\n"
            "Perintah yang tersedia:\n"
            "- /status - Cek status perangkat\n"
            "- /mode - Ganti mode (Manual/Otomatis)\n"
            "- /learn - (Admin) Latih ulang model\n\n"
            "Anda juga bisa memberi perintah langsung seperti 'nyalakan lampu'."
        )
        bot.reply_to(message, welcome_text, parse_mode="Markdown")

    elif command == 'mode':
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("🤖 Mode Otomatis", callback_data="set_mode:otomatis"),
            InlineKeyboardButton("👤 Mode Manual", callback_data="set_mode:manual")
        )
        bot.send_message(message.chat.id, f"Saat ini sistem dalam *Mode {current_mode.capitalize()}*.\n\nPilih mode baru:", reply_markup=markup, parse_mode="Markdown")

    elif command == 'status':
        try:
            lamp_status = "Menyala" if blynk.get_status(BLYNK_LAMPU_PIN) == "1" else "Mati"
            atap_status = "Terbuka" if blynk.get_status(BLYNK_ATAP_PIN) == "1" else "Tertutup"
            kipas_status = "Menyala" if blynk.get_status(BLYNK_KIPAS_PIN) == "1" else "Mati"
            status_msg = (
                f"🔄 *Status Perangkat Saat Ini*\n"
                f"--------------------------------------\n"
                f"💡 Lampu: *{lamp_status}*\n"
                f"🏠 Atap: *{atap_status}*\n"
                f"🌀 Kipas: *{kipas_status}*\n"
                f"⚙️ Mode Sistem: *{current_mode.capitalize()}*"
            )
            bot.reply_to(message, status_msg, parse_mode="Markdown")
        except Exception as e:
            bot.reply_to(message, f"❌ Gagal memeriksa status: {str(e)}")

    elif command == 'learn':
        if message.from_user.id != ADMIN_USER_ID:
            bot.reply_to(message, "⛔ Anda tidak memiliki akses untuk perintah ini.")
            return
        try:
            bot.reply_to(message, "🧠 Memulai proses pembelajaran...")
            if nlp.learn_from_log(AUTO_ENRICH_LOG_FILE, MODEL_DIR):
                nlp = IndonesianNLPPipelineNLTK.load_model(MODEL_DIR)
                nlp.intents_filepath = INTENTS_JSON_PATH
                bot.send_message(message.chat.id, "✅ Pembelajaran selesai! Model telah diperbarui.")
            else:
                bot.send_message(message.chat.id, "ℹ️ Tidak ada data baru yang unik untuk dipelajari.")
        except Exception as e:
            bot.send_message(message.chat.id, f"❌ Terjadi error saat proses pembelajaran: {str(e)}")

# --- HANDLER UNTUK TOMBOL INLINE (DENGAN PERBAIKAN) ---
@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    global current_mode
    parts = call.data.split(':')
    action = parts[0]
    device = parts[1]

    try:
        bot.answer_callback_query(call.id)

        if action == "set_mode":
            if device == "otomatis":
                blynk.send_command(BLYNK_MODE_PIN, 1)
                current_mode = "otomatis"
                bot.edit_message_text(f"✅ Sistem beralih ke *Mode Otomatis*.", call.message.chat.id, call.message.message_id, parse_mode="Markdown")
            elif device == "manual":
                blynk.send_command(BLYNK_MODE_PIN, 0)
                current_mode = "manual"
                bot.edit_message_text(f"✅ Sistem beralih ke *Mode Manual*.", call.message.chat.id, call.message.message_id, parse_mode="Markdown")
        
        elif action == "force_off":
            if device == "kipas":
                blynk.send_command(BLYNK_KIPAS_PIN, 0)
                bot.edit_message_text("✅ Oke, kipas telah dimatikan secara manual.", call.message.chat.id, call.message.message_id)
        
        elif action == "force_on":
            if device == "lampu":
                blynk.send_command(BLYNK_LAMPU_PIN, 1)
                bot.edit_message_text("✅ Oke, lampu telah dinyalakan secara manual.", call.message.chat.id, call.message.message_id)
            elif device == "kipas":
                blynk.send_command(BLYNK_KIPAS_PIN, 1)
                bot.edit_message_text("✅ Oke, kipas telah dinyalakan secara manual.", call.message.chat.id, call.message.message_id)

        elif action == "force_open":
            if device == "atap":
                blynk.send_command(BLYNK_ATAP_PIN, 1)
                bot.edit_message_text("✅ Oke, atap telah dibuka secara manual.", call.message.chat.id, call.message.message_id)

        elif action == "cancel":
            bot.edit_message_text("👍 Baik, perintah dibatalkan.", call.message.chat.id, call.message.message_id)

    except Exception as e:
        print(f"Error handling callback: {e}")

# --- HANDLER UTAMA UNTUK PESAN TEKS & SUARA ---
@bot.message_handler(content_types=['text', 'voice'])
def handle_message(message):
    try:
        text = ""
        if message.content_type == 'voice':
            text = convert_voice_to_text(message.voice)
            bot.reply_to(message, f"🔊 Anda berkata: {text}")
        else:
            text = message.text

        if text.startswith('/'): return

        result = nlp.predict(text, auto_enrich_log_file=AUTO_ENRICH_LOG_FILE)
        
        if 'low_confidence_guess' in result['intent']:
            bot.reply_to(message, result['response'])
            return
            
        if result['confidence'] > 0.65 and result['intent'] is not None:
            intent = result['intent']
            
            control_intents = ['nyala_lampu', 'matikan_lampu', 'buka_atap', 'tutup_atap', 'nyala_kipas', 'mati_kipas']
            if current_mode == "otomatis" and intent in control_intents:
                bot.reply_to(message, "⚠️ Perintah tidak dapat dijalankan.\nSistem sedang dalam *Mode Otomatis*.", parse_mode="Markdown")
                return

            # --- LOGIKA PERINTAH MODE MANUAL ---
            if intent == 'nyala_lampu':
                ldr_val_str = blynk.get_status(BLYNK_LDR_PIN)
                if ldr_val_str and float(ldr_val_str) < LDR_THRESHOLD_DARK:
                    markup = InlineKeyboardMarkup().row(InlineKeyboardButton("Tetap Nyalakan", callback_data="force_on:lampu"), InlineKeyboardButton("Batal", callback_data="cancel:lampu"))
                    bot.send_message(message.chat.id, "💡 Peringatan: Kondisi masih terang. Yakin ingin menyalakan lampu?", reply_markup=markup)
                    return
                if blynk.get_status(BLYNK_LAMPU_PIN) == "1": bot.send_message(message.chat.id, "💡 Lampu sudah menyala.")
                else: blynk.send_command(BLYNK_LAMPU_PIN, 1); bot.send_message(message.chat.id, "✅ Berhasil menyalakan lampu.")
            
            elif intent == 'matikan_lampu':
                if blynk.get_status(BLYNK_LAMPU_PIN) == "0": bot.send_message(message.chat.id, "💡 Lampu sudah mati.")
                else: blynk.send_command(BLYNK_LAMPU_PIN, 0); bot.send_message(message.chat.id, "✅ Berhasil mematikan lampu.")

            elif intent == 'buka_atap':
                rain_val_str = blynk.get_status(BLYNK_RAINDROP_PIN)
                if rain_val_str and float(rain_val_str) < RAIN_THRESHOLD_WET:
                    markup = InlineKeyboardMarkup().row(InlineKeyboardButton("Tetap Buka (Bahaya!)", callback_data="force_open:atap"), InlineKeyboardButton("Batal", callback_data="cancel:atap"))
                    bot.send_message(message.chat.id, "❌ PERINGATAN! Sensor mendeteksi HUJAN. Yakin ingin tetap membuka atap?", reply_markup=markup)
                    return
                if blynk.get_status(BLYNK_ATAP_PIN) == "1": bot.send_message(message.chat.id, "🏠 Atap sudah terbuka.")
                else: blynk.send_command(BLYNK_ATAP_PIN, 1); bot.send_message(message.chat.id, "✅ Berhasil membuka atap.")

            elif intent == 'tutup_atap':
                if blynk.get_status(BLYNK_ATAP_PIN) == "0": bot.send_message(message.chat.id, "🏠 Atap sudah tertutup.")
                else: blynk.send_command(BLYNK_ATAP_PIN, 0); bot.send_message(message.chat.id, "✅ Berhasil menutup atap.")

            elif intent == 'nyala_kipas':
                if blynk.get_status(BLYNK_KIPAS_PIN) == "1": bot.send_message(message.chat.id, "🌀 Kipas sudah menyala.")
                else: blynk.send_command(BLYNK_KIPAS_PIN, 1); bot.send_message(message.chat.id, "✅ Berhasil menyalakan kipas.")

            elif intent == 'mati_kipas':
                temp_val_str = blynk.get_status(BLYNK_DHT_TEMP_PIN)
                if temp_val_str and float(temp_val_str) > TEMP_THRESHOLD_HOT:
                    markup = InlineKeyboardMarkup().row(InlineKeyboardButton("Ya, Tetap Matikan", callback_data="force_off:kipas"), InlineKeyboardButton("Batal", callback_data="cancel:kipas"))
                    bot.send_message(message.chat.id, f"🌀 Peringatan: Suhu masih panas ({float(temp_val_str):.1f}°C). Yakin ingin mematikan kipas?", reply_markup=markup)
                    return
                if blynk.get_status(BLYNK_KIPAS_PIN) == "0": bot.send_message(message.chat.id, "🌀 Kipas sudah mati.")
                else: blynk.send_command(BLYNK_KIPAS_PIN, 0); bot.send_message(message.chat.id, "✅ Berhasil mematikan kipas.")
            
            # ### PERUBAHAN DI SINI ###
            elif intent == 'cek_suhu':
                temp = blynk.get_status(BLYNK_DHT_TEMP_PIN)
                if temp:
                    bot.send_message(message.chat.id, f"🌡️ Suhu saat ini: {temp}°C")
                else:
                    bot.send_message(message.chat.id, "❌ Gagal mengambil data suhu.")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        bot.reply_to(message, f"❌ Terjadi kesalahan tak terduga: {str(e)}")

if __name__ == "__main__":
    print("🤖 Bot sedang berjalan...")
    bot.polling(none_stop=True)
