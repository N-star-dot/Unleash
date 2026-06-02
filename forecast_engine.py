import torch
import torch.nn as nn
from collections import deque
import numpy as np

class BiometricLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(BiometricLSTM, self).__init__()
        # The LSTM layer processes the sequence
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # The linear layer outputs the future prediction
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # We only care about the prediction from the final time step
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Initialize and load pre-trained weights (assuming you trained this offline)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forecasting_model = BiometricLSTM().to(device)
# forecasting_model.load_state_dict(torch.load("unleash_lstm_weights.pth"))
forecasting_model.eval()

# A rolling buffer that automatically drops the oldest reading when full
sequence_length = 60 
live_biometric_buffer = deque(maxlen=sequence_length)

def update_buffer(current_hr, current_hrv):
    """Appends live telemetry. Returns a tensor if the buffer is full."""
    # We must scale the data (e.g., Min-Max scaling) for the neural network to converge efficiently
    scaled_hr = current_hr / 200.0   # Rough normalization
    scaled_hrv = current_hrv / 100.0 # Rough normalization
    
    live_biometric_buffer.append([scaled_hr, scaled_hrv])
    
    if len(live_biometric_buffer) == sequence_length:
        # Convert the buffer into a PyTorch tensor shaped: (1, seq_length, 2)
        tensor_input = torch.tensor(np.array(live_biometric_buffer), dtype=torch.float32).unsqueeze(0)
        return tensor_input.to(device)
    return None
