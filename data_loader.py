"""
Data loading and preprocessing pipeline for signal upsampling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
from typing import Tuple, Optional
import os


class IQSignalDataset(Dataset):
    """
    Dataset for loading IQ signal pairs (low-resolution and high-resolution)
    """
    
    def __init__(self, 
                 low_res_file: str,
                 high_res_file: str,
                 chunk_size: int = 265,
                 overlap: int = 0,
                 normalize: bool = True,
                 upsample_factor: int = 5):
        """
        Args:
            low_res_file: Path to low-resolution signal file
            high_res_file: Path to high-resolution signal file
            chunk_size: Size of chunks to extract
            overlap: Overlap between consecutive chunks
            normalize: Whether to normalize signals
            upsample_factor: Expected upsampling factor
        """
        self.low_res_file = low_res_file
        self.high_res_file = high_res_file
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.normalize = normalize
        self.upsample_factor = upsample_factor
        
        # Load signals
        print("Loading signals...")
        self.low_res_signal = self._load_signal(low_res_file)
        self.high_res_signal = self._load_signal(high_res_file)
        
        print(f"Low-res signal shape: {self.low_res_signal.shape}")
        print(f"High-res signal shape: {self.high_res_signal.shape}")
        
        # Verify upsampling factor
        ratio = len(self.high_res_signal) / len(self.low_res_signal)
        print(f"Upsampling factor: {ratio:.2f}")
        assert abs(ratio - upsample_factor) < 0.01, \
            f"Upsampling factor mismatch: expected {upsample_factor}, got {ratio:.2f}"
        
        # Normalize if requested
        if self.normalize:
            self.low_res_signal = self._normalize(self.low_res_signal)
            self.high_res_signal = self._normalize(self.high_res_signal)
            print("Signals normalized")
        
        # Create chunk indices
        self.chunk_indices = self._create_chunk_indices()
        print(f"Total chunks: {len(self.chunk_indices)}")
    
    # def _load_signal(self, filename: str) -> np.ndarray:
    #     """
    #     Load IQ signal from file
    #
    #     Args:
    #         filename: Path to signal file
    #
    #     Returns:
    #         Complex signal as numpy array of shape (num_samples, 2) where 2 = [I, Q]
    #     """
    #     samples = []
    #     with open(filename, 'r') as f:
    #         for line in f:
    #             parts = line.strip().split()
    #             if len(parts) == 2:
    #                 # Convert hex strings to signed 16-bit integers
    #                 # Convert hex string to unsigned integer
    #                 i_unsigned = int(parts[0], 16)
    #                 q_unsigned = int(parts[1], 16)
    #
    #                 # Handle 16-bit signed conversion (Two's Complement)
    #                 i_val = i_unsigned if i_unsigned < 0x8000 else i_unsigned - 0x10000
    #                 q_val = q_unsigned if q_unsigned < 0x8000 else q_unsigned - 0x10000
    #
    #                 samples.append([i_val, q_val])

        # return np.array(samples, dtype=np.float32)
    def _load_signal(self, filename: str) -> np.ndarray:
        """
        Load IQ signal from a decimal text file.

        Args:
            filename: Path to the decimal signal file (two columns: I and Q)

        Returns:
            Complex signal as numpy array of shape (num_samples, 2)
        """
        print(f"Loading decimal signals from: {filename}")

        # loadtxt handles lines, stripping, and whitespace splitting automatically
        # It converts decimal strings (e.g., "125.5" or "-32000") directly to float32
        try:
            samples = np.loadtxt(filename, dtype=np.float32)
        except ValueError as e:
            print(f"Error: Ensure the file contains only decimal numbers. Details: {e}")
            raise

        # Safety check: Ensure the output is always 2D (num_samples, 2)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 2)

        return samples

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal to zero mean and unit variance

        Args:
            signal: Input signal of shape (num_samples, 2)

        Returns:
            Normalized signal
        """
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (signal - mean) / std

    def _create_chunk_indices(self) -> list:
        """
        Create list of chunk start indices

        Returns:
            List of (low_res_idx, high_res_idx) tuples
        """
        indices = []
        stride = self.chunk_size - self.overlap

        # For low-res signal
        num_chunks = (len(self.low_res_signal) - self.chunk_size) // stride + 1

        for i in range(num_chunks):
            low_res_idx = i * stride
            high_res_idx = low_res_idx * self.upsample_factor

            # Verify we have enough samples in high-res signal
            if high_res_idx + self.chunk_size * self.upsample_factor <= len(self.high_res_signal):
                indices.append((low_res_idx, high_res_idx))

        return indices

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample pair (low-res, high-res)

        Args:
            idx: Index of the chunk

        Returns:
            Tuple of (low_res_chunk, high_res_chunk) as torch tensors
        """
        low_res_idx, high_res_idx = self.chunk_indices[idx]

        # Extract chunks
        low_res_chunk = self.low_res_signal[
            low_res_idx:low_res_idx + self.chunk_size
        ]

        high_res_chunk = self.high_res_signal[
            high_res_idx:high_res_idx + self.chunk_size * self.upsample_factor
        ]

        # Convert to tensors
        low_res_tensor = torch.from_numpy(low_res_chunk).float()
        high_res_tensor = torch.from_numpy(high_res_chunk).float()

        return low_res_tensor, high_res_tensor


class SignalDataModule:
    """
    Data module for managing train/val/test splits
    """

    def __init__(self,
                 low_res_file: str,
                 high_res_file: str,
                 batch_size: int = 32,
                 chunk_size: int = 256,
                 overlap: int = 0,
                 normalize: bool = True,
                 upsample_factor: int = 5,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 num_workers: int = 4):
        """
        Args:
            low_res_file: Path to low-resolution signal file
            high_res_file: Path to high-resolution signal file
            batch_size: Batch size for data loaders
            chunk_size: Size of chunks to extract
            overlap: Overlap between consecutive chunks
            normalize: Whether to normalize signals
            upsample_factor: Expected upsampling factor
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            num_workers: Number of workers for data loading
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create dataset
        self.dataset = IQSignalDataset(
            low_res_file=low_res_file,
            high_res_file=high_res_file,
            chunk_size=chunk_size,
            overlap=overlap,
            normalize=normalize,
            upsample_factor=upsample_factor
        )

        # Split dataset
        total_size = len(self.dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

        print(f"\nDataset split:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading pipeline...\n")

    data_module = SignalDataModule(
        #low_res_file='/home/ubuntu/upload/iq_in_iq_60.txt',
        #high_res_file='/home/ubuntu/upload/iq_in_iq_300.txt',
        low_res_file=r"C:\Users\ziyadfahoum\Desktop\int prj\iqdatazizo10.60.txt",
        high_res_file=r"C:\Users\ziyadfahoum\Desktop\int prj\iqdatazizo10.300.txt",
        batch_size=8,
        chunk_size=256,
        overlap=0,
        normalize=True,
        upsample_factor=5
    )
    
    # Test train dataloader
    train_loader = data_module.train_dataloader()
    print(f"\nTrain dataloader batches: {len(train_loader)}")
    
    # Get first batch
    low_res, high_res = next(iter(train_loader))
    print(f"\nFirst batch shapes:")
    print(f"  Low-res: {low_res.shape}")
    print(f"  High-res: {high_res.shape}")
    
    assert low_res.shape[0] == 8, "Batch size mismatch"
    assert low_res.shape[1] == 256, "Chunk size mismatch"
    assert low_res.shape[2] == 2, "Complex dimension mismatch"
    assert high_res.shape[1] == 256 * 5, "High-res chunk size mismatch"
    
    print("\n✓ Data loading test passed!")
