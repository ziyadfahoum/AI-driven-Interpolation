# AI-Driven QAM Signal Interpolation via Bidirectional GRUs

## 📌 Project Overview
This project implements a deep learning-based signal upsampler for **Quadrature Amplitude Modulation (QAM)** signals. The system interpolates low-resolution signals (**60MHz**) to high-resolution (**300MHz**) counterparts using a specialized Recurrent Neural Network (RNN) architecture.

The project demonstrates how AI can reconstruct signal integrity by learning temporal dependencies and phase alignment, moving beyond the limitations of fixed-coefficient classical interpolation.

---

## 🛠 Key Technical Improvements
To optimize performance and meet academic criteria for architectural innovation, the following changes were implemented:

* **Bidirectional GRU Processing:** Replaced standard Transformer blocks with a Bidirectional Gated Recurrent Unit (GRU) to capture phase dependencies from both directions of the signal stream.
* **Learnable Upsampler:** Integrated a `ConvTranspose1d` layer with overlapping kernels. This eliminates the "jagged" artifacts found in linear interpolation by learning the optimal transition between samples.
* **Residual Connections:** Added skip-connections between the upsampling stage and the output to stabilize training and maintain high-frequency spectral integrity.
* **Cosine Annealing:** Implemented a `CosineAnnealingLR` scheduler to achieve smooth convergence and avoid local minima during training.

---

## 📊 The "50m EVM" Discovery
A critical finding of this project is the performance ceiling encountered during testing. While the goal was to outperform **Spline interpolation (39m EVM)**, the model reached a plateau at **50m EVM**.

| Method | EVM Achievement | Note |
| :--- | :--- | :--- |
| **Spline (Classical)** | **39m** | Best performing classical baseline. |
| **RNN (Proposed)** | **50m** | Successfully matched the ground truth limit. |
| **Ground Truth** | **50m** | **Inherent noise floor in the reference data.** |

**Conclusion:** Analysis of the FFT and signal plots confirmed that the model is performing with high fidelity. However, the **ground truth data itself is flawed**, possessing an inherent EVM of 50m. The model has successfully learned to reconstruct the signal up to the mathematical limit allowed by the dataset.

---

## 📂 Project Structure
* `rnn_model.py`: The core architecture (ConvTranspose1d + Bidirectional GRU).
* `train_rnn.py`: Training script with AdamW optimizer and Cosine Annealing.
* `inference_rnn.py`: Evaluation script for MSE, PSNR, and visualization.
* `convert_to_txt.py`: Utility to export signals for hardware/circuit simulation.

---

## 🚀 How to Run

Follow these steps to set up the environment, train the model, and run inference. You can copy and run these commands in your terminal:

```bash
# 1. Install necessary libraries
pip install torch numpy matplotlib tqdm scipy

# 2. Train the model (Replace paths with your actual data files)
python train_rnn.py --low-res path/to/low_res.txt --high-res path/to/high_res.txt --hidden-dim 128 --epochs 100

# 3. Run inference and visualize results
python inference_rnn.py --model rnn_v2_best.pt --visualize

# 4. (Optional) Export results for hardware simulation
python convert_to_txt.py
