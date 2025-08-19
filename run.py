from src.train import train_model
from src.dataset import prepare_dataset
from src.evaluate import evaluate_model
from config.config import MODEL_PATH

CSV_PATH = "data/IMDB Dataset.csv"

if __name__ == "__main__":
    print("🔹 Starting training...")
    model, history = train_model(CSV_PATH)

    print("🔹 Preparing dataset for evaluation...")
    X_train, X_test, y_train, y_test = prepare_dataset(CSV_PATH)

    print("🔹 Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("✅ Done! Model saved at:", MODEL_PATH)
