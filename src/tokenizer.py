import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from config.config import VOCAB_SIZE, MAX_LEN, TOKENIZER_PATH

def create_tokenizer(texts):
    # Create and adapt vectorizer
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LEN
    )
    vectorizer.adapt(texts)

    # Save vectorizer
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer

def encode_texts(vectorizer, texts):
    # Apply vectorizer directly
    sequences = vectorizer(texts)
    return sequences.numpy()   # convert tensor -> numpy array

# # Example usage
# texts = [
#     "This movie was amazing and full of surprises!",
#     "I hated the film, it was boring and predictable.",
#     "The acting was good but the story was too long.",
#     "An excellent thriller with great visuals and music.",
#     "Worst movie ever! Waste of time and money.",
#     "Just okay, nothing special about this one.",
#     "Loved the suspense and the ending twist!",
#     "Bad direction, weak script, but nice soundtrack.",
#     "Fantastic! Totally worth watching twice.",
#     "Average film with decent acting but poor editing."
# ]


# vectorizer = create_tokenizer(texts)
# encoded = encode_texts(vectorizer, texts)
# print(encoded.shape)   # (10, MAX_LEN)
# print(encoded[:3])     # show first 3 sequences
