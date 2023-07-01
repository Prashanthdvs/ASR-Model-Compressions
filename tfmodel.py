#Convert ASR model to TFLite

import tensorflow as tf
from transformers import TFWhisperModel

model = TFWhisperModel.from_pretrained(r"C:\Users\damojipurapuv.d\Downloads\flask offline\whisper_medium")
tflite_model_path = 'whisper.tflite'



converter = tf.lite.TFLiteConverter.from_saved_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# For normal conversion:
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]

# For conversion with FP16 quantization:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True

# For conversion with hybrid quantization:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.experimental_new_converter = True

tflite_model = converter.convert()

