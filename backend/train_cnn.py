import tensorflow as tf
from model.cnn_model import build_cnn_model
import config


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

    model = build_cnn_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=64, verbose=1)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    model.save(config.CNN_MODEL_PATH)
    print(f"Saved CNN model to {config.CNN_MODEL_PATH}")
    print(f"Test accuracy: {acc:.4f}")
