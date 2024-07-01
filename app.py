import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from joblib import load
from tensorflow.keras.models import load_model
import pandas as pd

# Load scaler, encoder, dan model
scaler = load('model/scaler.pkl')
encoder = load('model/label_encoder.pkl')
model = load_model('model/best_model.h5')

EXPECTED_FEATURE_LENGTH = 17496  # Panjang fitur yang diharapkan
AUDIO_FILE = 'temp_audio.wav'  # Path file audio sementara

# Custom CSS color untuk tampilan aplikasi
st.markdown(
    """
    <style>
    .stApp {
        background-color: #01012b;
    }
    .emotion-label {
        font-size: 24px;
        font-weight: bold;
        color: #33ff33;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Pemetaan emosi ke emoji
emotion_emoji_dict = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😃",
    "sad": "😔",
    "angry": "😡",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😮"
}

# Fungsi untuk mengklasifikasi emosi dari file audio
def emotion_classifier(file_path, language='English'):
    class FeatureExtractor:
        def __init__(self, frame_length=2048, hop_length=512):
            self.frame_length = frame_length
            self.hop_length = hop_length

        # Ekstraksi fitur Zero Crossing Rate
        def zcr(self, data):
            return librosa.feature.zero_crossing_rate(data, frame_length=self.frame_length, hop_length=self.hop_length).flatten()

        # Ekstraksi fitur Root Mean Square Energy
        def rmse(self, data):
            return librosa.feature.rms(y=data, frame_length=self.frame_length, hop_length=self.hop_length).flatten()

        # Ekstraksi fitur MFCC
        def mfcc(self, data, sr, n_mfcc=13, flatten=True):
            mfcc_features = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=self.hop_length)
            return mfcc_features.T.flatten() if flatten else mfcc_features.T

        # Ekstraksi fitur Chroma
        def chroma(self, data, sr):
            chroma_features = librosa.feature.chroma_stft(y=data, sr=sr, hop_length=self.hop_length)
            return chroma_features.T.flatten()

        # Ekstraksi fitur Spectral Contrast
        def spectral_contrast(self, data, sr):
            contrast_features = librosa.feature.spectral_contrast(y=data, sr=sr, hop_length=self.hop_length)
            return contrast_features.T.flatten()

        # Ekstraksi fitur Mel Spectrogram
        def mel_spectrogram(self, data, sr):
            mel_features = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=self.hop_length)
            return librosa.power_to_db(mel_features).flatten()

        # Ekstraksi semua fitur yang dibutuhkan dari audio
        def extract_features(self, data, sr):
            zcr_features = self.zcr(data)
            rmse_features = self.rmse(data)
            mfcc_features = self.mfcc(data, sr)
            chroma_features = self.chroma(data, sr)
            spectral_contrast_features = self.spectral_contrast(data, sr)
            mel_spectrogram_features = self.mel_spectrogram(data, sr)
            return np.concatenate([zcr_features, rmse_features, mfcc_features, chroma_features, spectral_contrast_features, mel_spectrogram_features])

    class DataAugmentation:
        @staticmethod
        # Penambahan noise ke data audio
        def noise(data, noise_factor=0.005):
            noise_amp = noise_factor * np.random.uniform() * np.amax(data)
            return data + noise_amp * np.random.normal(size=data.shape[0])

        @staticmethod
        # Perubahan pitch data audio
        def pitch(data, sr, n_steps=4):
            return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

    class AudioProcessor:
        def __init__(self, frame_length=2048, hop_length=512):
            self.feature_extractor = FeatureExtractor(frame_length, hop_length)
            self.augmenter = DataAugmentation()

        # Mendapatkan fitur dari audio dengan durasi dan offset tertentu
        def get_features(self, path, duration=2.5, offset=0.6):
            try:
                # Memuat data audio dari path yang diberikan dengan durasi dan offset yang ditentukan
                data, sr = librosa.load(path, duration=duration, offset=offset)
            except Exception as e:
                st.error(f"Error loading audio file: {e}")
                return None
            # Mengekstrak fitur dari data audio asli
            features = [self.feature_extractor.extract_features(data, sr)]

            # Menambahkan noise ke data audio dan mengekstrak fiturnya
            noised_audio = self.augmenter.noise(data)
            features.append(self.feature_extractor.extract_features(noised_audio, sr))

            # Mengubah pitch data audio dan mengekstrak fiturnya
            pitched_audio = self.augmenter.pitch(data, sr)
            features.append(self.feature_extractor.extract_features(pitched_audio, sr))

            # Menambahkan noise ke data audio yang telah diubah pitch-nya dan mengekstrak fiturnya
            pitched_noised_audio = self.augmenter.noise(pitched_audio)
            features.append(self.feature_extractor.extract_features(pitched_noised_audio, sr))

            return np.array(features)

    processor = AudioProcessor()
    X = processor.get_features(file_path)

    if X is None:
        return None, None

    try:
        # Jika panjang fitur tidak sesuai dengan yang diharapkan, lakukan penyesuaian
        if X.shape[1] != EXPECTED_FEATURE_LENGTH:
            if X.shape[1] < EXPECTED_FEATURE_LENGTH:
                X = np.pad(X, ((0, 0), (0, EXPECTED_FEATURE_LENGTH - X.shape[1])), mode='constant')
            else:
                X = X[:, :EXPECTED_FEATURE_LENGTH]

        # Melakukan skala fitur dan mereshape untuk dimasukkan ke model
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1, 1))

        # Memprediksi emosi dari fitur yang telah disiapkan
        predictions = model.predict(X_scaled)
        predicted_emotion = encoder.inverse_transform(np.argmax(predictions, axis=1))
    except ValueError as e:
        st.error(f"Error in feature scaling or reshaping: {e}")
        return None, None

    return predicted_emotion[0], predictions[0]

# Fungsi untuk memprediksi emosi dari file yang diunggah
def predict(model, file_uploaded):
    with open(AUDIO_FILE, 'wb') as f:
        f.write(file_uploaded.read())
    emotion, _ = emotion_classifier(AUDIO_FILE)
    return emotion

# Fungsi untuk merekam audio
def record_audio(duration=10, fs=44100):
    st.write(f"Recording dalam {duration} detik...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(AUDIO_FILE, audio, fs)
    st.write("Recording selesai.")
    return AUDIO_FILE

# Fungsi untuk menangani input audio dari pengguna
def handle_audio_input():
    models_load_state = st.text('Loading models...')
    models_load_state.text('Models Loading Complete!')

    st.sidebar.title("Made by Kelompok 5C")  # Judul sidebar
    st.sidebar.text("""
1) Putu Yuki Parmawati (2208561066)
2) Ni Luh Gede Cahaya Putri Mahadewi (2208561110)
3) Made Pranajaya Dibyacita (2208561122)
4) Amsal Hamonangan Butarbutar (2208561134)
    """)  # Informasi anggota kelompok
    st.sidebar.markdown("[GitHub Repository](https://github.com/mdprana/Speech-Emotion-Recognition-PPDM)")  # Link ke repositori GitHub

    # Opsi untuk memilih metode prediksi
    option = st.sidebar.radio("Pilih Opsi Prediksi:", ["Upload Audio", "Record Audio"])

    # Jika pengguna memilih untuk mengunggah audio
    if option == "Upload Audio":
        files_uploaded = st.file_uploader("Pilih file audio...", type=['wav', 'mp3'], accept_multiple_files=True)
        
        if files_uploaded:
            # Jika hanya satu file yang diunggah
            if len(files_uploaded) == 1:
                file = files_uploaded[0]
                st.audio(file, format=f"audio/{file.name.split('.')[-1]}")
                emotion = predict(model, file)
                if emotion:
                    emoji = emotion_emoji_dict.get(emotion, "")
                    st.markdown(f'<div class="emotion-label">Prediksi Emosi: {emotion} {emoji}</div>', unsafe_allow_html=True)
                else:
                    st.error('Failed to predict emotion for the uploaded audio.')
            else:
                emotions = []
                # Jika beberapa file diunggah, prediksi emosi untuk masing-masing file
                for idx, file in enumerate(files_uploaded):
                    st.audio(file, format=f"audio/{file.name.split('.')[-1]}")
                    emotion = predict(model, file)
                    if emotion:
                        emoji = emotion_emoji_dict.get(emotion, "")
                        emotions.append({'No': idx + 1, 'Nama File': file.name, 'Prediksi Emosi': f'{emotion} {emoji}'})
                    else:
                        st.error(f"Failed to predict emotion for {file.name}.")
                
                if emotions:
                    emotion_df = pd.DataFrame(emotions)
                    emotion_df.index = np.arange(1, len(emotion_df) + 1)
                    st.table(emotion_df.set_index('No'))

    # Jika pengguna memilih untuk merekam audio
    elif option == "Record Audio":
        duration = st.slider("Pilih durasi recording (detik)", min_value=1, max_value=10, value=5)
        if st.button("Start Recording"):
            record_audio(duration)
            st.audio(AUDIO_FILE, format='audio/wav')
            data, sr = librosa.load(AUDIO_FILE)
            plt.figure(figsize=(14, 5))
            librosa.display.waveshow(data, sr=sr)
            plt.title('Waveform')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            st.pyplot(plt)
            emotion, _ = emotion_classifier(AUDIO_FILE)
            if emotion:
                emoji = emotion_emoji_dict.get(emotion, "")
                st.markdown(f'<div class="emotion-label">Emosi pada Record Audio adalah: {emotion} {emoji}</div>', unsafe_allow_html=True)
            else:
                st.error('Gagal memprediksi emosi pada Record Audi.')

# Fungsi utama untuk menjalankan aplikasi
def main():
    st.image('speechrecognition.webp')
    st.title('Speech Emotion Recognition')
    handle_audio_input()

if __name__ == "__main__":
    main()
