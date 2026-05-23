from tensorflow.keras.datasets import reuters
import numpy as np
import os

vocab_size = 10000

os.makedirs("data", exist_ok=True)

(x_train, y_train), (x_val, y_val) = reuters.load_data(
    num_words=vocab_size
)

np.save("data/x_train.npy", x_train, allow_pickle=True)
np.save("data/y_train.npy", y_train, allow_pickle=True)

np.save("data/x_val.npy", x_val, allow_pickle=True)
np.save("data/y_val.npy", y_val, allow_pickle=True)

print("Dataset preprocessing completed")