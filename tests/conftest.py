import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Portable CPU unit tests; Metal integration is tested with the CLI training run.
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
