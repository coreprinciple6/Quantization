# Quantization
Learning about Quantization from a fantastic course by [DeepLearning.ai](https://learn.deeplearning.ai/accomplishments/2561fbb7-ff71-499b-a05f-f7b9af9a45e1?usp=sharing)

### Quantization in practice by applying QAT method on Speech Recognition Task using LibriSpeech and wav2vec model

1. Data Preparation:
    - Download and preprocess the LibriSpeech dataset. Using only subset of 1800hrs for training
    - Implement data augmentation techniques (speed perturbation, SpecAugment).
2. Model Setup:
    - Load pre-trained Wav2Vec 2.0 model from Hugging Face.
    - Implement a custom PyTorch module for the full speech recognition pipeline. No extra layers added to the model yet.
    - model_name = "facebook/wav2vec2-base-960h"
    - Purpose of using Processor: The Wav2Vec2Processor combines two important components:
        - A feature extractor (Wav2Vec2FeatureExtractor): This processes raw audio inputs.
        - A tokenizer (Wav2Vec2CTCTokenizer): This handles text inputs and outputs.
        - It prepares raw audio inputs for the model by performing necessary preprocessing steps like resampling, normalization, and padding.
        - It also handles the conversion between text and token IDs, which is crucial for tasks like speech recognition where you need to map between audio and text.
3. Quantization-Aware Training:
    - Modify the model to include fake quantization nodes. Didnt implement yet
    - Implement custom quantization configurations for different parts of the model (e.g., convolutional layers vs. attention layers).
        - used qnnpack(which is optimized for mobile devices.typically uses symmetric quantization) on feature extractor layers and fbgemm( which is optimized for server-side inference on x86 processors.generally used for asymmetric quantization.) on encoder layers
    - Finetune the model with quantization awareness, fine-tuning on LibriSpeech.
    - training parameters
        - num_epochs = 2
        - optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        - batch size: 4
        - scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        - criterion = nn.CTCLoss(zero_infinity=True)
