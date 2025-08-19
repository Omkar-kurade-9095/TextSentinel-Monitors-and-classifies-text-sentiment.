import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.dataset import prepare_dataset
from src.model import build_model
from config.config import MODEL_PATH, EPOCHS, BATCH_SIZE

def train_model(csv_path):
    X_train, X_test, y_train, y_test = prepare_dataset(csv_path)
    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return model, history
