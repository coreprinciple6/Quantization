from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor#AutoProcessor
import torch.nn as nn
import torch.quantization as quant

# Custom PyTorch module for the full speech recognition pipeline
class SpeechRecognitionModel(nn.Module):
    def __init__(self, wav2vec2_model):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        # Add any additional layers or modifications here

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        return outputs.logits

def qat_prep_model():

    speech_model, processor = wav2vec_specs()
    # Define quantization configuration
    print("Preparing for Quantization-Aware Training...")
    qconfig = quant.get_default_qat_qconfig('qnnpack')
    speech_model.qconfig = qconfig
    # Prepare the model for QAT
    speech_model = quant.prepare_qat(speech_model)
    return speech_model, processor

def wav2vec_specs():
    # Load pre-trained Wav2Vec 2.0 model and processor
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)#AutoProcessor.from_pretrained(model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True)
    # Initialize the custom model
    speech_model = SpeechRecognitionModel(wav2vec2_model)

    return speech_model, processor