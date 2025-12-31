# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('imdb_rnn_model.h5')  # ✅ updated model filename

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    # Ensure any word not in the word_index is replaced with index 2 (for 'unknown')
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # `2` is used for out-of-vocabulary words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit Page Config and Styling
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: rgb(32,33,39);
            color: white;
        }
        .stTextArea textarea {
            background-color: #2c2f35;
            color: white;
            border: 1px solid rgb(24,239,199);
            border-radius: 6px;
        }
        .stButton>button {
            background-color: rgb(24,239,199);
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: white;
        }
        .stTextInput, .stTextArea {
            color: white;
        }
        .stWarning, .stInfo, .stSuccess {
            background-color: rgba(24, 239, 199, 0.3);
            color: white;
        }
        .stMarkdown {
            color: rgb(24, 239, 199);
        }
        /* Placeholder styling */
        .stTextArea textarea::placeholder {
            color: rgba(255, 255, 255, 0.5);  /* Softer color for placeholder */
        }
    </style>
""", unsafe_allow_html=True)

# Step 4: App UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.markdown("Enter a movie review and find out if it's **positive** or **negative**!", unsafe_allow_html=True)

# Text input
user_input = st.text_area('📝 Movie Review', placeholder="Type your review here...", height=200)

# Classify button
if st.button('🚀 Classify'):
    if user_input.strip() == "":
        st.warning("Please enter a review to classify.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = '🟢 Positive' if prediction[0][0] > 0.5 else '🔴 Negative'

        st.subheader(f"Sentiment: {sentiment}")
        st.caption(f"Prediction Confidence: {prediction[0][0]:.2f}")
else:
    st.info("Enter a review above and click **Classify** to see the sentiment.")
