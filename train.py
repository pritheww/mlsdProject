# Importing Libraries and Parameters as per Requirement 7

import keras
from keras import ops
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import mlflow
import mlflow.keras


vocab_size = 10000
maxlen = 200
embed_dim = 32
num_heads = 4
ff_dim1 = 32
ff_dim2 = 64

mlflow.start_run()

mlflow.log_param("vocab_size", vocab_size)
mlflow.log_param("maxlen", maxlen)
mlflow.log_param("embed_dim", embed_dim)
mlflow.log_param("num_heads", num_heads)
mlflow.log_param("ff_dim1", ff_dim1)
mlflow.log_param("ff_dim2", ff_dim2)

x_train = np.load("data/x_train.npy", allow_pickle=True)
y_train = np.load("data/y_train.npy", allow_pickle=True)

x_val = np.load("data/x_val.npy", allow_pickle=True)
y_val = np.load("data/y_val.npy", allow_pickle=True)


x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

print(f"Training shape: {x_train.shape}")

# As per Text Classification with Transformer example in Keras
# this implementation uses only the encoder part of the Transformer architecture.
# the number of encoder layers is controlled using the variable `num_transformer_layers`
# as required, and the model is evaluated for three different depths.


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # ff_dim is set dynamically
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
def build_sandwich_model(num_layers):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

# as per requirement 3,4,6 regarding the intermediate blocks
    # First encoder block (ff_dim = 32)
    #requirement 5
    x = TransformerBlock(embed_dim, num_heads, ff_dim=ff_dim1)(x)

    # We subtract 2 because we manually add the first and last blocks.
    num_intermediate = num_layers - 2

    # iterative as per requirement 6

    for _ in range(num_intermediate):
        # As per requirement 5 ff_dim is 64 (ff_dim2) for intermediate blocks
        x = TransformerBlock(embed_dim, num_heads, ff_dim=ff_dim2)(x)

    #requirement 5
    # Final encoder block (ff_dim = 32)
    x = TransformerBlock(embed_dim, num_heads, ff_dim=ff_dim1)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(46, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

layer_settings = [3, 5, 7]
results = {}

for n_layers in layer_settings:
    print(f"\nTraining Model with {n_layers} Transformer Layers")

    # Build model
    model = build_sandwich_model(n_layers)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=10, #requirment 7
        validation_data=(x_val, y_val),
        verbose=1
    )


    y_pred_probs = model.predict(x_val)
    y_pred = np.argmax(y_pred_probs, axis=1)


    f1 = f1_score(y_val, y_pred, average='weighted')
    
    mlflow.log_metric(f"f1_score_{n_layers}_layers", float(f1))

    val_accuracy = float(history.history['val_accuracy'][-1])
    train_accuracy = float(history.history['accuracy'][-1])

    mlflow.log_metric(f"train_accuracy_{n_layers}_layers", train_accuracy)
    mlflow.log_metric(f"val_accuracy_{n_layers}_layers", val_accuracy)


    results[n_layers] = {
        'model': model,
        'history': history,
        'f1': f1,
        'y_pred': y_pred
    }

    # requirment 8
    print(f"Displaying Confusion Matrix for {n_layers} layers:")
    cm = confusion_matrix(y_val, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='viridis', include_values=False)
    plt.title(f"Confusion Matrix: {n_layers} Layers")
    plt.show()

# requirment 9
best_setting = max(results, key=lambda k: results[k]['f1'])
print(f"\nConclusion: The model with {best_setting} layers performed best with F1-score: {results[best_setting]['f1']:.4f}")

# bar chart comparison as per requirment 10
# Scores from previous assignment (hardcoded)
previous_assignment_scores = {
    "Simple RNN": 0.2451,
    "LSTM": 0.4793,
    "GRU": 0.6155,
    "BiSimpleRNN": 0.6308,
    "BiLSTM": 0.7634,
    "BiGRU": 0.7083
}


all_scores = previous_assignment_scores.copy()
all_scores[f"Transf (3 layers)"] = results[3]['f1']
all_scores[f"Transf (5 layers)"] = results[5]['f1']
all_scores[f"Transf (7 layers)"] = results[7]['f1']

# Plotting
names = list(all_scores.keys())
values = list(all_scores.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(names, values, color=['gray']*6 + ['blue', 'blue', 'blue'])


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom')

plt.title("F1 Score Comparison: Previous Models vs Transformer Variants")
plt.ylabel("Weighted F1 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import os

os.makedirs("model", exist_ok=True)

model.save("model/model.keras")
mlflow.keras.log_model(model, "model")

mlflow.end_run()

