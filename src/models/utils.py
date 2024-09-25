import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_data(X):
    scaler = MinMaxScaler()
    X_flattened = X.view(X.size(0), -1)
    X_scaled = torch.tensor(scaler.fit_transform(X_flattened.cpu()), dtype=torch.float32).to(device)
    return X_scaled.view(X.size())

def validate(model, validation_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(validation_loader)

class CNN(nn.Module):
    def __init__(self, input_size, in_channels=8):
        super(CNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 12, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(12, 24, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))

        # Dynamically determine the flattened size
        self.flattened_size = self._initialize_flattened_size(input_size, in_channels)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(24, 2)

    def _initialize_flattened_size(self, input_size, in_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_size, 2)
            out = self.conv1(dummy_input)
            out = self.pool(self.relu(out))
            out = self.conv2(out)
            out = self.pool(self.relu(out))
            flattened_size = out.view(1, -1).size(1)
            #print(f"Flattened size: {flattened_size}")  # Debug print for flattened size
            return flattened_size

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(self.relu(out))
        out = self.conv2(out)
        out = self.pool(self.relu(out))
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
        #print(f"Shape before fc1: {out.shape}")  # Debug print
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(GRUModel, self).__init__()
        # GRU Layers conforme as especificações de entrada e saída
        self.gru1 = nn.GRU(input_size, 64, 1, batch_first=True)  # GRU Layer 1
        self.dropout1 = nn.Dropout(0.3)                         # Dropout após GRU Layer 1
        self.gru2 = nn.GRU(64, 128, 1, batch_first=True)        # GRU Layer 2
        self.dropout2 = nn.Dropout(0.3)                         # Dropout após GRU Layer 2
        self.gru3 = nn.GRU(128, 256, 1, batch_first=True)       # GRU Layer 3

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully connected layers conforme as especificações
        self.fc1 = nn.Linear(256, 128)                          
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        # Forward pass através das GRU Layers e Dropouts
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = self.dropout2(out)
        out, _ = self.gru3(out)

        # Flatten output
        out = self.flatten(out)

        # Forward pass através das camadas lineares e ReLU
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        
        return out


def train_model(input_positions, delta_input, delta_output, model_class, epochs=15, batch_size=8, learning_rate=0.03):
    n_samples = input_positions.shape[0]
    x_size = input_positions.shape[1]

    # Prepare training tensors with the correct shape
    X_train = torch.zeros((n_samples, 8, x_size, 2)).to(device)  # Adjust in_channels to 8
    y_train = torch.zeros((n_samples, 2)).to(device)

    for sample_idx in range(n_samples):
        input_pos = input_positions[sample_idx]
        X_train[sample_idx, :, :, :] = torch.tensor(input_pos, dtype=torch.float32).expand(8, x_size, 2)  # Expand to match in_channels
        y_train[sample_idx, 0] = input_pos[-1, 0] + delta_output
        y_train[sample_idx, 1] = input_pos[-1, 1] + delta_output

    X_train = normalize_data(X_train)
    # Split data into training and validation sets
    train_ratio = 0.8
    n_train = int(len(X_train) * train_ratio)
    n_val = len(X_train) - n_train

    # Training and validation datasets
    train_data = X_train[:n_train]
    val_data = X_train[n_train:]
    train_labels = y_train[:n_train]
    val_labels = y_train[n_train:]

    # PyTorch datasets
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    # PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model based on the provided class
    if model_class in [LSTMModel, GRUModel]:
        model = model_class(input_size=2, hidden_size=64, num_layers=2, output_size=2).to(device)
    else:
        model = model_class(x_size).to(device)  # For CNNModel

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate(model, validation_loader, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return model



def Making_Predictions(input_positions, delta_input, delta_output, model, model_type='cnn', device='cuda'):
    """
    Function to make predictions using the given model (CNN, LSTM, or GRU).

    Parameters:
    - input_positions: numpy array of input data
    - delta_input: time difference between input samples
    - delta_output: time difference between the last input and the output
    - model: trained model to be used for prediction
    - model_type: type of model ('cnn', 'lstm', or 'gru')
    - device: device to run the model on ('cuda' or 'cpu')

    Returns:
    - out_pos: numpy array with predicted positions
    """
    n_samples = input_positions.shape[0]
    out_pos = np.zeros((n_samples, 2))

    x_size = input_positions.shape[1]
    time_steps = delta_input * np.arange(x_size)  # Time vector for inputs
    model.eval()  # Set model to evaluation mode

    for sample_idx in tqdm(range(n_samples), desc='Estimating input positions'):
        input_pos = input_positions[sample_idx]
        final_time = time_steps[-1] + delta_output  # Adjust the final time considering delta_output

        # Adjust input for model type
        if model_type == 'cnn':
    # Convert input to 4D for CNN (batch size, channels, height, width)
            inputs = torch.tensor(input_pos, dtype=torch.float32).unsqueeze(0).repeat(1, 8, 1, 1).to(device)  # Repeat across channel dimension to make 8 channels
        elif model_type in ['lstm', 'gru']:
            # Convert input to 3D for RNN (batch size, sequence length, input size)
            inputs = torch.tensor(input_pos, dtype=torch.float32).unsqueeze(0).to(device)

        # Generate prediction using the model
        prediction = model(inputs)

        # Move prediction to CPU and convert to a numpy array
        out_pos[sample_idx, :] = prediction.detach().cpu().numpy().flatten()

        # Adjust predictions based on delta_output
        out_pos[sample_idx, 0] += delta_output * (final_time / time_steps[-1])
        out_pos[sample_idx, 1] += delta_output * (final_time / time_steps[-1])

    return out_pos




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 1), stride=stride, padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        # Ensure the residual and output have the same dimensions before adding
        if residual.shape != out.shape:
            residual = nn.functional.interpolate(residual, size=out.shape[2:], mode='nearest')

        out += residual
        out = self.relu(out)
        return out

class Enhanced2DCNN(nn.Module):
    def __init__(self, input_size):
        super(Enhanced2DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, downsample=nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(64)
            )),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1)),
            nn.Dropout(0.4)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128)
            )),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1)),
            nn.Dropout(0.4)
        )

        # Determine the flattened size dynamically
        self._initialize_flattened_size(input_size)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def _initialize_flattened_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size, 2).to(device)  # Exemplo de entrada tensorial no dispositivo correto
            out = self.layer1(dummy_input)
            out = self.layer2(out)
            out = self.layer3(out)
            self.flattened_size = out.view(1, -1).size(1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = self.fc4(out)
        return out
