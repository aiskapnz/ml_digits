import tensorflow as tf

# In case of "NameError: name 'tensorflow' is not defined":
# comment out the "exec("del tensorflow")" in
# the file ".../site-packages/openvino/frontend/tensorflow/utils.py" at line 93, in get_environment_setup

keras_model = tf.keras.models.load_model("./models/tf_learn_digits.keras")
keras_model.export("./models/tf_learn_digits.xml", format="openvino")
