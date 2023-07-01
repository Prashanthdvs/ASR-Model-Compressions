from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from onnxruntime.quantization import QuantFormat, QuantizationMode
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration
import os
from functools import partial
from optimum.onnxruntime.configuration import AutoCalibrationConfig


# create ORTQuantizer and define quantization configuration
quantizer = ORTQuantizer.from_pretrained('staticquantfiles')
qconfig = AutoQuantizationConfig.avx512_vnni(
    is_static=True,
    format=QuantFormat.QOperator,
    mode=QuantizationMode.QLinearOps,
    per_channel=True,
    operators_to_quantize=["MatMul", "Add" ]
    )


def preprocess_fn(ex, tokenizer):
    return tokenizer(ex["text"],padding="longest")

# Create the calibration dataset
calibration_samples = 256
calibration_dataset = quantizer.get_calibration_dataset(
    dataset_name=r'C:\Users\damojipurapuv.d\Downloads\audio test\audio test',
    #preprocess_function=partial(preprocess_fn, tokenizer=WhisperProcessor.from_pretrained('Shubham09/LISA_Whisper_medium_latest')
)
    #num_samples=calibration_samples,
    #dataset_split="train",


# Create the calibration configuration containing the parameters related to calibration.
calibration_config = AutoCalibrationConfig.percentiles(calibration_dataset, percentile=99.99239080907178)

# Perform the calibration step: computes the activations quantization ranges
shards=16
for i in range(shards):
    shard = calibration_dataset.shard(shards, i)
    quantizer.partial_fit(
        dataset=shard,
        calibration_config=calibration_config,
        onnx_model_path="./staticmodel.onnx",
        operators_to_quantize=qconfig.operators_to_quantize,
        batch_size=calibration_samples//shards,
        use_external_data_format=False,
    )
ranges = quantizer.compute_ranges()

# remove temp augmented model again
os.remove("augmented_model.onnx")

