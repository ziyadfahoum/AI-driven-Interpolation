"""
Inference script for RNN signal upsampling model (Matched to V2 logic)
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt

# --- Minimal Change: Import the RNN model ---
from rnn_model import RNNSignalUpsampler

class SignalUpsampler:
    def __init__(self,
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 hidden_dim: int = 128,  # Minimal Change: Match default
                 num_layers: int = 4,    # Minimal Change: Added dynamic layers
                 upsample_factor: int = 5):

        self.device = torch.device(device)
        self.upsample_factor = upsample_factor

        # --- Minimal Change: Initialize RNN model dynamically ---
        self.model = RNNSignalUpsampler(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,       # Minimal Change: Now uses argument
            upsample_factor=upsample_factor
        ).to(self.device)

        # --- Minimal Change: Load raw state_dict ---
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        print(f"  RNN Model loaded from {model_path}")
        print(f"Using device: {self.device}")

    def upsample_signal(self,
                        signal: np.ndarray,
                        batch_size: int = 64) -> np.ndarray:
        signal = signal.astype(np.float32)
        num_samples = len(signal)
        upsampled_chunks = []

        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                chunk = signal[start_idx:end_idx]

                if len(chunk) < batch_size:
                    pad_size = batch_size - len(chunk)
                    chunk = np.pad(chunk, ((0, pad_size), (0, 0)), mode='edge')

                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)

                upsampled_chunk = self.model(chunk_tensor).squeeze(0).cpu().numpy()

                actual_len = (end_idx - start_idx) * self.upsample_factor
                upsampled_chunks.append(upsampled_chunk[:actual_len])

        return np.vstack(upsampled_chunks)

    def _load_signal(self, filename: str) -> np.ndarray:
        print(f"Loading signals from: {filename}")
        samples = np.loadtxt(filename, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 2)
        return samples

    def compute_mse(self, upsampled, reference):
        min_len = min(len(upsampled), len(reference))
        return np.mean((upsampled[:min_len] - reference[:min_len]) ** 2)

    def compute_psnr(self, upsampled, reference):
        mse = self.compute_mse(upsampled, reference)
        max_val = np.max(np.abs(reference))
        return 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')


# --- Visualization Function (UNTOUCHED to ensure identical plots) ---
def visualize_upsampling(low_res, upsampled, reference, output_path='upsampling_result_rnn.png', samples_to_plot=500):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    low_res_time = np.arange(min(samples_to_plot, len(low_res)))
    upsampled_time = np.arange(min(samples_to_plot * 5, len(upsampled)))
    ref_time = np.arange(min(samples_to_plot * 5, len(reference)))

    # I Component
    axes[0, 0].plot(low_res_time, low_res[:samples_to_plot, 0], 'b-', label='Low-res')
    axes[0, 1].plot(upsampled_time, upsampled[:samples_to_plot * 5, 0], 'r-', label='RNN Upsampled')
    axes[0, 1].plot(ref_time, reference[:samples_to_plot * 5, 0], 'g--', alpha=0.7, label='Reference')

    # Q Component
    axes[1, 0].plot(low_res_time, low_res[:samples_to_plot, 1], 'b-')
    axes[1, 1].plot(upsampled_time, upsampled[:samples_to_plot * 5, 1], 'r-')
    axes[1, 1].plot(ref_time, reference[:samples_to_plot * 5, 1], 'g--', alpha=0.7)

    # Magnitude & Error
    upsampled_mag = np.abs(upsampled[:samples_to_plot * 5, 0] + 1j * upsampled[:samples_to_plot * 5, 1])
    ref_mag = np.abs(reference[:samples_to_plot * 5, 0] + 1j * reference[:samples_to_plot * 5, 1])
    axes[2, 0].plot(upsampled_time, upsampled_mag, 'r-', label='Upsampled')
    axes[2, 0].plot(ref_time, ref_mag, 'g--', label='Reference')

    error = upsampled[:samples_to_plot * 5] - reference[:samples_to_plot * 5]
    axes[2, 1].plot(upsampled_time, np.abs(error[:, 0] + 1j * error[:, 1]), 'k-')

    for ax in axes.flatten(): ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Visualization saved to {output_path}")
    plt.show()


def main():
    # Base path added for clean default arguments
    base_path = '/content/drive/MyDrive/DEEP LEARNING PROJECT/'
    
    parser = argparse.ArgumentParser()
    # Changed default names to match your V2 requirements and direct to your folder
    parser.add_argument('--model', type=str, default=base_path + 'rnn_v2_best.pt')
    parser.add_argument('--low-res', type=str, default=base_path + 'iqdatazizo20.60.txt')
    parser.add_argument('--high-res', type=str, default=base_path + 'iqdatazizo20.300.txt')
    parser.add_argument('--output', type=str, default=base_path + 'upsample_signal_v2.npy')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--hidden-dim', type=int, default=128) # Default adjusted
    parser.add_argument('--num-layers', type=int, default=4)   # Added num-layers flag
    args = parser.parse_args()

    # Minimal Change: Pass both architecture parameters down
    upsampler = SignalUpsampler(
        model_path=args.model, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers
    )
    
    low_res = upsampler._load_signal(args.low_res)
    reference = upsampler._load_signal(args.high_res)

    upsampled = upsampler.upsample_signal(low_res)

    # I increased the formatting slightly to .8f to capture the 10^-6 precision
    print(f"MSE: {upsampler.compute_mse(upsampled, reference):.8f}")
    print(f"PSNR: {upsampler.compute_psnr(upsampled, reference):.2f} dB")

    np.save(args.output, upsampled)
    print(f"  Array saved to {args.output}")

    if args.visualize:
        # Save plot right next to the .npy file
        plot_path = base_path + 'upsampling_result_v2.png'
        visualize_upsampling(low_res, upsampled, reference, output_path=plot_path)

if __name__ == "__main__":
    main()