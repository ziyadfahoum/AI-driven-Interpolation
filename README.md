# AI-driven-Interpolation
Project Overview
This repository contains the implementation of a deep learning-based signal upsampler designed for Quadrature Amplitude Modulation (QAM) signals. The project focuses on interpolating low-resolution signals (60MHz) to high-resolution (300MHz) counterparts using a specialized Recurrent Neural Network (RNN) architecture.

Developed by Ziyad and Munir , a final-year Electrical Engineering students, this project was born from a collaboration with Mohamed Khamsi to explore the intersection of AI and high-speed hardware communication.

Key Features & Architectural Improvements
To meet the "significant change" requirements for academic validation, this project evolved from a standard Transformer-based approach to a highly optimized Bidirectional GRU framework:

Learnable Upsampler: Replaced static linear interpolation with a ConvTranspose1d layer. This allows the model to learn overlapping context, eliminating the "jagged" artifacts typical of classical methods.

Temporal Processing: Utilizes a Bidirectional Gated Recurrent Unit (GRU) to capture both forward and backward temporal dependencies, which is critical for maintaining phase alignment in I/Q signals.

Residual Learning: Implemented a skip-connection (residual) between the upsampler and the final projection layer to stabilize deep training and preserve signal integrity.

Advanced Optimization: Uses Cosine Annealing learning rate scheduling to ensure smooth convergence toward global minima.

Analysis: While the goal was to achieve an EVM lower than 39m, the model plateaued at 50m. Detailed Error Vector Magnitude (EVM) and FFT analysis revealed that the ground truth signal itself is flawed, possessing an inherent noise floor of 50m. The model successfully learned to reconstruct the signal with high fidelity relative to the provided reference, but it cannot mathematically surpass the quality of its own training data.

# Repository Structure
rnn_model.py: Defines the RNNSignalUpsampler class, including the feature extractor, transpose convolution upsampler, and bidirectional GRU layers.

train_rnn.py: The training pipeline featuring AdamW optimization, gradient clipping, and Cosine Annealing schedulers.

inference_rnn.py: A robust script for model evaluation, calculating MSE, PSNR, and generating visual upsampling plots.

convert_to_txt.py: A utility script to handle high-precision data conversion for hardware simulation compatibility.
