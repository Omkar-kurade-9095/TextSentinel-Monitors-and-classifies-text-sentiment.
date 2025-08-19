# TextSentinel

**TextSentinel** is a web-based application that monitors and classifies text sentiment.  
It allows users to input text (e.g., movie reviews) and predicts whether the sentiment is **Positive** or **Negative** using a deep learning model (LSTM).  

---

## 🚀 Features

- Classifies text sentiment in real-time.
- Clean and preprocess input text (removes HTML, URLs, special characters).
- Uses tokenization and padding for sequence modeling.
- LSTM-based neural network for accurate sentiment prediction.
- Streamlit web app interface for easy user interaction.
- Displays results with friendly emojis for quick understanding.

---


---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/TextSentinel.git
cd TextSentinel

2. Create a virtual environment (optional but recommended):

python -m venv myenv
source myenv/bin/activate      # Linux / Mac
myenv\Scripts\activate         # Windows

3. Install dependencies:
pip install -r requirements.txt

🏃 Running the Project
1. Train the model (if not already trained)

python run.py

Saves trained model to saved_models/model_best.h5.

Saves tokenizer to saved_models/tokenizer.pkl.

2. Run the Streamlit web app

streamlit run app.py

Open browser at: http://localhost:8501

Input text to see sentiment prediction.

🐳 Running with Docker

1. Build Docker image:

docker build -t textsentinel-app .

2. Run container:

docker run -p 8501:8501 textsentinel-app

Open browser at: http://localhost:8501


📊 Model Details

Architecture: LSTM-based neural network

Input: Preprocessed text, tokenized and padded

Output: Binary sentiment classification (Positive / Negative)

Training Dataset: IMDB Movie Reviews

Preprocessing: HTML removal, lowercasing, special character removal




🛠 Tech Stack

Python: Data processing & modeling

TensorFlow / Keras: Deep learning

Pandas / NumPy: Data manipulation

Streamlit: Web app interface

Docker: Containerization (optional)
