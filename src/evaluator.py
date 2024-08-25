import torch
import os
import time
from tqdm import tqdm
from jiwer import wer
from data_preprocess import prepare_data
from model import wav2vec_specs

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using MPS for acceleration.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")

def calculate_wer(model, dataloader, processor, device):
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch_waveforms, batch_transcripts in tqdm(dataloader, desc="Calculating WER"):
            batch_waveforms = batch_waveforms.to(device)
            outputs = model(batch_waveforms)
            predicted_ids = torch.argmax(outputs, dim=-1)
            predicted_transcripts = processor.batch_decode(predicted_ids)
            
            all_predictions.extend(predicted_transcripts)
            all_references.extend(batch_transcripts)
    
    return wer(all_references, all_predictions)

def measure_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")
    return size_mb

def measure_inference_speed(model, dataloader, device, num_runs=100):
    model.eval()
    latencies = []
    total_samples = 0
    total_time = 0
    
    with torch.no_grad():
        for _ in range(num_runs):
            for batch_waveforms, _ in dataloader:
                batch_waveforms = batch_waveforms.to(device)
                batch_size = batch_waveforms.size(0)
                
                start_time = time.time()
                _ = model(batch_waveforms)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.extend([latency] * batch_size)
                
                total_samples += batch_size
                total_time += (end_time - start_time)
                
                if total_samples >= num_runs:
                    break
            if total_samples >= num_runs:
                break
    
    avg_latency = sum(latencies) / len(latencies)
    throughput = total_samples / total_time
    
    return avg_latency, throughput

def main():
    # Load data
    print("Loading test data...")
    _, test_loader = prepare_data(train_duration=0, test_duration=600)  # Only load test data

    # Load the saved model
    print("Loading the saved quantized model...")
    model, processor = wav2vec_specs()
    # Load the state dict
    model.load_state_dict(torch.load("checkpoints/quantized_speech_recognition_model.pth", map_location=device))
    model.eval()

    # Evaluate the model
    print("Evaluating the model...")
    
    # Word Error Rate
    wer_score = calculate_wer(model, test_loader, processor, device)
    print(f"Word Error Rate: {wer_score:.4f}")

    # Model Size
    model_size = measure_model_size(model)
    print(f"Quantized Model Size: {model_size:.2f} MB")

    # Inference Speed
    avg_latency, throughput = measure_inference_speed(model, test_loader, device)
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"Inference Throughput: {throughput:.2f} samples/second")

    print("Evaluation complete!")

if __name__ == "__main__":
    main()