import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import CustomObjectScope

# Load the pre-trained model
@st.cache_resource
def load_emotion_model():
    with CustomObjectScope({}):
        return load_model('model/model.h5')

model = load_emotion_model()

def preprocess_audio(file):
    """Load and preprocess the audio file."""
    try:
        audio, sr = librosa.load(file, sr=22050)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None, None

def extract_features(data, sr):
    """Extract features from the audio."""
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))
    return result

def predict_emotion(features):
    """Predict emotion from audio features."""
    if features is not None:
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        predicted_emotion = np.argmax(prediction)
        confidence_score = np.max(prediction)
        return predicted_emotion, confidence_score
    else:
        return None, None

def emotion_label(emotion_index):
    """Map the prediction index to emotion labels."""
    labels = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fearful', 7: 'Disgust', 8: 'Surprised'}
    return labels.get(emotion_index, 'Unknown')

def plot_waveform(audio, sr):
    """Plot the audio waveform."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    st.pyplot(plt)

def plot_mfccs(mfccs, sr):
    """Plot MFCCs."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit App
st.title("Speech Emotion Recognition")
st.write("Upload an audio file to predict the emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    try:
        st.audio(uploaded_file, format='audio/wav')
        audio, sr = preprocess_audio(uploaded_file)
        if audio is not None:
            plot_waveform(audio, sr)
            features = extract_features(audio, sr)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr)
            plot_mfccs(mfccs, sr)
            emotion_index, confidence_score = predict_emotion(features)
            if emotion_index is not None:
                emotion = emotion_label(emotion_index)
                st.write(f"Predicted Emotion: **{emotion}** with confidence {confidence_score:.2f}")
            else:
                st.error("Could not predict emotion.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
