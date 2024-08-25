import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using MPS for acceleration.")
else:
    device = torch.device("cpu")
    print("MPS device not found. Falling back to CPU.")

class LibriSpeechSubset(Dataset):
    def __init__(self, root_dir, subset="train-clean-100", transform=None, max_duration=3600):
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.max_duration = max_duration  # maximum duration in seconds
        self.data = []
        self._load_data()

    def _load_data(self):
        total_duration = 0
        for speaker_id in os.listdir(self.root_dir):
            speaker_dir = os.path.join(self.root_dir, speaker_id)
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                if not os.path.isdir(chapter_dir):
                    continue
                for file in os.listdir(chapter_dir):
                    if file.endswith(".flac"):
                        audio_path = os.path.join(chapter_dir, file)
                        duration = librosa.get_duration(filename=audio_path)
                        if total_duration + duration > self.max_duration:
                            return
                        total_duration += duration
                        transcript_path = os.path.join(chapter_dir,file[:-10] + ".trans.txt") 
                        with open(transcript_path, "r") as f:
                            transcript = f.read().strip()
                        self.data.append((audio_path, transcript))
        print(f"Total duration of loaded data: {total_duration:.2f} seconds")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, transcript = self.data[idx]
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return torch.from_numpy(waveform).float(), transcript

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def time_stretch(self, waveform, rate=1.0):
        return librosa.effects.time_stretch(waveform, rate=rate)

    def pitch_shift(self, waveform, n_steps):
        return librosa.effects.pitch_shift(waveform, sr=self.sample_rate, n_steps=n_steps)

    def add_noise(self, waveform, noise_factor=0.005):
        noise = np.random.randn(len(waveform))
        return waveform + noise_factor * noise

    def __call__(self, waveform):
        # Randomly apply augmentations
        if np.random.rand() < 0.5:
            waveform = self.time_stretch(waveform, rate=np.random.uniform(0.9, 1.1))
        if np.random.rand() < 0.5:
            waveform = self.pitch_shift(waveform, n_steps=np.random.uniform(-2, 2))
        if np.random.rand() < 0.5:
            waveform = self.add_noise(waveform)
        
        return waveform

def collate_fn(batch):
    waveforms, transcripts = zip(*batch)
    waveforms = [w.float() for w in waveforms]
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return waveforms, transcripts

def prepare_data(root_dir="./data", batch_size=8, train_duration=1800, test_duration=600):
    print("Creating custom datasets with augmentation...")
    augmentation = AudioAugmentation()
    train_data = LibriSpeechSubset(root_dir, subset="LibriSpeech/train-clean-100", transform=augmentation, max_duration=train_duration)
    test_data = LibriSpeechSubset(root_dir, subset="LibriSpeech/test-clean", max_duration=test_duration)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    return train_loader, test_loader


# if __name__ == "__main__":
#     train_loader, test_loader = prepare_data(train_duration=1800, test_duration=600)  # 30 minutes train, 10 minutes test
#     print(f"Number of training batches: {len(train_loader)}")
#     print(f"Number of test batches: {len(test_loader)}")

#     # Test the data loader
#     for batch_waveforms, batch_transcripts in train_loader:
#         batch_waveforms = batch_waveforms.to(device)
#         print(f"Batch waveforms shape: {batch_waveforms.shape}")
#         print(f"Sample transcript: {batch_transcripts[0]}")
#         break