import streamlit as st
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from src.tokenizer import encode_texts
from config.config import MAX_LEN, MODEL_PATH, TOKENIZER_PATH

st.set_page_config(page_title="Text Classification App", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Classifier")

# Debug info
st.write("🔄 Loading model & tokenizer...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"⚠️ Model not loaded: {e}")
    model = None

try:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    st.success("✅ Tokenizer loaded successfully")
except Exception as e:
    st.error(f"⚠️ Tokenizer not loaded: {e}")
    tokenizer = None

user_input = st.text_area("✍️ Enter your review here:")

if st.button("Predict"):
    if not model or not tokenizer:
        st.error("❌ Model or tokenizer not available. Train first with: python run.py")
    elif user_input.strip() == "":
        st.warning("⚠️ Please enter some text for prediction.")
    else:
        seq = encode_texts(tokenizer, [user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        pred = model.predict(padded)[0][0]
        prediction = "Positive 😀" if pred > 0.5 else "Negative 😡"
        st.success(f"Prediction: **{prediction}**")

st.write("✅ UI loaded successfully")
