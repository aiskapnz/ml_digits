from pathlib import Path

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=10)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

path = Path("./models")
path.mkdir(parents=True, exist_ok=True)
probability_model.save(f"{path.absolute()}/tf_digits.keras")

# Converting to openvino model

# In case of "NameError: name 'tensorflow' is not defined":
# comment out the "exec("del tensorflow")" in
# the file ".../site-packages/openvino/frontend/tensorflow/utils.py" at line 93, in get_environment_setup

probability_model.export(f"{path.absolute()}/ov_digits.xml", format="openvino")
