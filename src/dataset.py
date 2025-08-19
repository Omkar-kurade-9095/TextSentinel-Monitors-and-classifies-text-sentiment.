import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import clean_text
from src.tokenizer import create_tokenizer, encode_texts
from config.config import TEST_SIZE, RANDOM_STATE

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df["review"] = df["review"].apply(clean_text)

    # Tokenizer
    tokenizer = create_tokenizer(df["review"].values)
    X = encode_texts(tokenizer, df["review"].values)
    y = df["sentiment"].map({"positive":1, "negative":0}).values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test



# csv_path="data/IMDB Dataset.csv"

# print(prepare_dataset(csv_path))