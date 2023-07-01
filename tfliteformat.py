import tensorflow as tf
from datasets import load_dataset
from transformers import AutoProcessor, TFWhisperForConditionalGeneration, GenerationConfig, WhisperProcessor
from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdirname:
    print('created temporary directory', tmpdirname)

def bin_to_tflite(repo_name):
    
    # Creating force_token_map to be used in GenerationConfig
    force_token_map = [[50258, 50266], [50359, 50363]] #

    # Creating generation_config with force_token_map
    generation_config = GenerationConfig(force_token_map=force_token_map)

    # Creating an instance of AutoProcessor from the pretrained model
    processor = WhisperProcessor.from_pretrained(repo_name,from_pt=True)

    # Creating an instance of TFWhisperForConditionalGeneration from the pretrained model
    model = TFWhisperForConditionalGeneration.from_pretrained(repo_name, from_pt=True)

    # Loading dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # Inputs
    inputs = processor(ds[0]["audio"]["array"], return_tensors="tf")
    input_features = inputs.input_features

    # Creating a GenerateModel Class
    class GenerateModel(tf.Module):
        def __init__(self, model):
            super(GenerateModel, self).__init__()
            self.model = model

        @tf.function(
            input_signature=[
            tf.TensorSpec(shape=(1, 80,3000), dtype=tf.float32, name="input_ids"),
            ]
        )
        def serving(self, input_ids):
            outputs = self.model.generate(input_ids,generation_config=generation_config) # forced_decoder_ids=force_token_map)
            return {"sequences": outputs}

    # Saving the model
    saved_model_dir = tmpdirname+'/'+'tf1'
    generate_model = GenerateModel(model=model)
    tf.saved_model.save(generate_model, saved_model_dir, signatures={"serving_default": generate_model.serving})

    # Converting to TFLite model
    tflite_model_path = repo_name +'_.tflite'
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    # Saving the TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    blobconnectstr='DefaultEndpointsProtocol=https;AccountName=stlisamed001;AccountKey=+y8XwkkgP4Dw1H/6+WnqN/n6fC5HleikDVeyNMIqNM7FgVllOZ0XArUOwijoK98XRMENCkRVPBoreCjS8yzp9A==;EndpointSuffix=core.windows.net'                                            
    blob_service_client = BlobServiceClient.from_connection_string(blobconnectstr)
    blob_client = blob_service_client.get_blob_client(container='offlinespeech2text', blob='compressed model/tflite/' + tflite_model_path)
    with open(tflite_model_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        print("uploading file ---->", tflite_model_path)    
    if tflite_model:
        return "Tflite generated sucessfully"
