# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from preprocess import prepare_data
# from model import qat_prep_model

# def text_to_indices(processor, texts):
#     return processor(text=texts, return_tensors="pt", padding=True).input_ids

# def train_epoch(model, dataloader, optimizer, criterion, scheduler, processor, device, accumulation_steps=4):
#     model.train()
#     total_loss = 0
#     optimizer.zero_grad()
#     for i, (batch_waveforms, batch_transcripts) in enumerate(tqdm(dataloader, desc="Training")):
#         batch_waveforms = batch_waveforms.to(device)
        
#         outputs = model(batch_waveforms)
        
#         # Convert transcripts to tensor of indices
#         target_indices = text_to_indices(processor, batch_transcripts).to(device)
        
#         # Move outputs and target_indices to CPU for CTC loss calculation
#         outputs_cpu = outputs.cpu()
#         target_indices_cpu = target_indices.cpu()
        
#         # Compute CTC loss on CPU
#         input_lengths = torch.full(size=(outputs_cpu.shape[0],), fill_value=outputs_cpu.shape[1], dtype=torch.long)
#         target_lengths = torch.sum(target_indices_cpu != processor.tokenizer.pad_token_id, dim=1)
#         loss = criterion(outputs_cpu.transpose(0, 1), target_indices_cpu, input_lengths, target_lengths)
        
#         # Normalize the loss to account for batch accumulation
#         loss = loss / accumulation_steps
#         loss.backward()

#         if (i + 1) % accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()
        
#         total_loss += loss.item() * accumulation_steps
    
#     scheduler.step()
#     return total_loss / len(dataloader)

# def evaluate(model, dataloader, criterion, processor, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for batch_waveforms, batch_transcripts in tqdm(dataloader, desc="Evaluating"):
#             batch_waveforms = batch_waveforms.to(device)
#             outputs = model(batch_waveforms)
            
#             # Convert transcripts to tensor of indices
#             target_indices = text_to_indices(processor, batch_transcripts).to(device)
            
#             # Move outputs and target_indices to CPU for CTC loss calculation
#             outputs_cpu = outputs.cpu()
#             target_indices_cpu = target_indices.cpu()
            
#             # Compute CTC loss on CPU
#             input_lengths = torch.full(size=(outputs_cpu.shape[0],), fill_value=outputs_cpu.shape[1], dtype=torch.long)
#             target_lengths = torch.sum(target_indices_cpu != processor.tokenizer.pad_token_id, dim=1)
#             loss = criterion(outputs_cpu.transpose(0, 1), target_indices_cpu, input_lengths, target_lengths)
            
#             total_loss += loss.item()
#     return total_loss / len(dataloader)

# def main():
#     # Check if MPS is available
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("MPS device found. Using MPS for acceleration.")
#     else:
#         device = torch.device("cpu")
#         print("No GPU found. Using CPU.")

#     # Load data with a smaller batch size
#     print("Loading data...")
#     batch_size = 4  # Reduced batch size
#     train_loader, test_loader = prepare_data(batch_size=batch_size, train_duration=1800, test_duration=600)

#     # Load model
#     print("Loading model...")
#     model, processor = qat_prep_model()
#     model = model.to(device)

#     # Training setup
#     num_epochs = 2
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
#     criterion = nn.CTCLoss()

#     # Gradient accumulation steps
#     accumulation_steps = 8  # This will simulate a batch size of 4 * 8 = 32

#     # Training loop
#     print("Starting Quantization-Aware Training...")
#     for epoch in range(num_epochs):
#         train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, processor, device, accumulation_steps)
#         val_loss = evaluate(model, test_loader, criterion, processor, device)
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     # Convert the model to a fully quantized version
#     print("Converting to fully quantized model...")
#     quantized_model = torch.quantization.convert(model.eval(), inplace=False)

#     # Save the quantized model
#     print("Saving quantized model...")
#     torch.save(quantized_model.state_dict(), "checkpoints/quantized_speech_recognition_model.pth")

# if __name__ == "__main__":
#     main()

''' ORIGINAL UP'''
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocess import prepare_data
from model import qat_prep_model

def text_to_indices(processor, texts):
    return processor(text=texts, return_tensors="pt", padding=True).input_ids

def train_epoch(model, dataloader, optimizer, criterion, scheduler, processor, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (batch_waveforms, batch_transcripts) in enumerate(tqdm(dataloader, desc="Training")):
        batch_waveforms = batch_waveforms.to(device)
        
        outputs = model(batch_waveforms)
        
        # Convert transcripts to tensor of indices
        target_indices = text_to_indices(processor, batch_transcripts).to(device)
        
        # Move outputs and target_indices to CPU for CTC loss calculation
        outputs_cpu = outputs.cpu()
        target_indices_cpu = target_indices.cpu()
        
        # Compute CTC loss on CPU
        input_lengths = torch.full(size=(outputs_cpu.shape[0],), fill_value=outputs_cpu.shape[1], dtype=torch.long)
        target_lengths = torch.sum(target_indices_cpu != processor.tokenizer.pad_token_id, dim=1)
        
        # Add a small epsilon to prevent log(0)
        outputs_cpu = outputs_cpu.clamp(min=1e-7, max=1 - 1e-7)
        
        # Ensure no zero-length targets
        target_lengths = target_lengths.clamp(min=1)
        
        loss = criterion(outputs_cpu.transpose(0, 1), target_indices_cpu, input_lengths, target_lengths)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {i}. Skipping this batch.")
            continue
        
        # Normalize the loss to account for batch accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    scheduler.step()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, processor, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_waveforms, batch_transcripts in tqdm(dataloader, desc="Evaluating"):
            batch_waveforms = batch_waveforms.to(device)
            outputs = model(batch_waveforms)
            
            # Convert transcripts to tensor of indices
            target_indices = text_to_indices(processor, batch_transcripts).to(device)
            
            # Move outputs and target_indices to CPU for CTC loss calculation
            outputs_cpu = outputs.cpu()
            target_indices_cpu = target_indices.cpu()
            
            # Compute CTC loss on CPU
            input_lengths = torch.full(size=(outputs_cpu.shape[0],), fill_value=outputs_cpu.shape[1], dtype=torch.long)
            target_lengths = torch.sum(target_indices_cpu != processor.tokenizer.pad_token_id, dim=1)
            
            # Add a small epsilon to prevent log(0)
            outputs_cpu = outputs_cpu.clamp(min=1e-7, max=1 - 1e-7)
            
            # Ensure no zero-length targets
            target_lengths = target_lengths.clamp(min=1)
            
            loss = criterion(outputs_cpu.transpose(0, 1), target_indices_cpu, input_lengths, target_lengths)
            
            if torch.isnan(loss):
                print("NaN loss detected during evaluation. Skipping this batch.")
                continue
            
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found. Using MPS for acceleration.")
    else:
        device = torch.device("cpu")
        print("No GPU found. Using CPU.")

    # Load data with a smaller batch size
    print("Loading data...")
    batch_size = 4  # Reduced batch size
    train_loader, test_loader = prepare_data(batch_size=batch_size, train_duration=1800, test_duration=600)

    # Load model
    print("Loading model...")
    model, processor = qat_prep_model()
    model = model.to(device)

    # Training setup
    num_epochs = 1
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CTCLoss(zero_infinity=True)

    # Gradient accumulation steps
    accumulation_steps = 8  # This will simulate a batch size of 4 * 8 = 32

    # Training loop
    print("Starting Quantization-Aware Training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, processor, device, accumulation_steps)
        val_loss = evaluate(model, test_loader, criterion, processor, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Convert the model to a fully quantized version
    print("Converting to fully quantized model...")
    quantized_model = torch.quantization.convert(model.eval(), inplace=False)

    # Save the quantized model
    print("Saving quantized model...")
    torch.save(quantized_model.state_dict(), "checkpoints/quantized_speech_recognition_model.pth")

if __name__ == "__main__":
    main()
