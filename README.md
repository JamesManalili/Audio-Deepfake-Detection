# Audio Deepfake Detection Model

A deep learning model that detects synthetic (deepfake) audio with 98.65% accuracy, achieving 98.60% precision and 98.76% recall.

## Model Architecture
**Hybrid CNN-LSTM Network**:
- 3 Convolutional Layers (32, 64, 128 filters) with BatchNorm and MaxPooling
- TimeDistributed Dense layer (64 units)
- LSTM layer (64 units)
- Dense classifier (128 units + Dropout)
- Softmax output (2 classes)

## Key Features
- Mel-spectrogram audio representation (64 bins)
- Audio augmentation (time stretch, pitch shift, noise injection)
- Class weighting for imbalanced data
- Mixed precision training (FP16)
- Multiprocessing for fast data loading
- Early stopping and learning rate reduction

## Dataset
- Dataset size: 176,000 audio files
- Binary classification: Real vs Fake audio
- Format: 16kHz, 5-second clips (.wav/.mp3)
- 
## Training Details
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 32
- Epochs: 20 (with early stopping)
- Regularization: L2 (λ=0.01), Dropout (0.5)

## Performance Metrics
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.65% |
| Precision | 98.60% |
| Recall    | 98.76% |

![image](https://github.com/user-attachments/assets/6921c22a-b15a-4ce1-a050-1a3657f8d30b)
