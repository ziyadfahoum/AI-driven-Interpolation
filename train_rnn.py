import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from rnn_model import RNNSignalUpsampler
from data_loader import IQSignalDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RNNSignalUpsampler(
        input_dim=2, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers, 
        upsample_factor=5,
        dropout=args.dropout
    ).to(device)
    
    # --- ADDED: Print Total Trainable Parameters ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Total Trainable Parameters: {total_params:,} ===\n")

    # ==========================================
    # CHECKPOINT LOADING BLOCK
    # ==========================================
    checkpoint_path = "rnn_v2_best.pt"
    if os.path.exists(checkpoint_path):
        print(f"✓ Existing checkpoint found! Loading {checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        except Exception as e:
            print(f"⚠ Architecture mismatch. Cannot load {checkpoint_path}. Starting fresh.")
    else:
        print(f"⚠ No checkpoint found at {checkpoint_path}. Training from scratch.")

    dataset = IQSignalDataset(args.low_res, args.high_res)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    best_val_error = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} | Current LR: {curr_lr:.2e}")

        for low, high in tqdm(train_loader, desc="Training"):
            low, high = low.to(device), high.to(device)
            
            optimizer.zero_grad()
            output = model(low)
            loss = criterion(output, high)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for low, high in val_loader:
                low, high = low.to(device), high.to(device)
                output = model(low)
                total_val_loss += criterion(output, high).item()
        
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train err: {avg_train:.9f} | Val err: {avg_val:.9f}")
        
        scheduler.step()

        if avg_val < best_val_error:
            best_val_error = avg_val
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ NEW BEST VAL ERROR! Saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-res", type=str, required=True)
    parser.add_argument("--high-res", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    train(parser.parse_args())