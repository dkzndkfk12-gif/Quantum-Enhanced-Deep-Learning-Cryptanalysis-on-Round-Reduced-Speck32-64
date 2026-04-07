import os
os.environ["OMP_NUM_THREADS"] = "1"

import copy
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import Speck1


def training_data_gen():
    if Speck1.check_testvector():
        print("Speck test vector matches, the Speck1 module is working correctly!\n")
    else:
        print("Speck test vector did NOT match, check your code!\n")

    n_samples = 128000
    rounds = 5
    X, Y = Speck1.make_train_data(n_samples, rounds, diff=(0x0040, 0))

    print("Data generation completed.")
    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
    print("Number of real (label=1) samples:", np.sum(Y == 1))
    print("Number of random (label=0) samples:", np.sum(Y == 0))
    return X, Y


X, Y = training_data_gen()

X_train, X_val, X_test = X[:7680, :], X[7680:10240, :], X[10240:, :]
Y_train, Y_val, Y_test = Y[:7680], Y[7680:10240], Y[10240:]

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

X_train = X_train.reshape(X_train.shape[0], 4, 16)
X_test = X_test.reshape(X_test.shape[0], 4, 16)
X_val = X_val.reshape(X_val.shape[0], 4, 16)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


# ----------------------------
# 논문 구조에 맞춘 4-qubit filter
# ----------------------------
NUM_QUBITS = 4
NUM_LAYERS = 4
NUM_FILTERS = 4
HIDDEN_DIM = 64

dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_filter(inputs, weights):
    # threshold encoding: 0 -> RX(0), 1 -> RX(pi)
    for i in range(NUM_QUBITS):
        qml.RX(inputs[i], wires=i)

    # 4-layer ansatz
    for layer in range(NUM_LAYERS):
        for i in range(NUM_QUBITS):
            qml.RX(weights[layer, i], wires=i)
        for i in range(NUM_QUBITS):
            qml.RZ(weights[layer, 4 + i], wires=i)

        qml.CNOT(wires=[3, 2])
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[1, 0])

    return qml.expval(qml.PauliZ(0))


def he_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class QCNNClassifier(nn.Module):
    def __init__(self, num_filters=NUM_FILTERS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.num_filters = num_filters

        self.qweights = nn.Parameter(0.05 * torch.randn(num_filters, NUM_LAYERS, 8))

        self.fc1 = nn.Linear(num_filters * 16, hidden_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # 0/1 binary -> 0/pi angle encoding
        cols = (x * np.pi).transpose(1, 2).contiguous().view(-1, 4)
        # cols shape = (batch * 16, 4)

        filter_outputs = []
        for f in range(self.num_filters):
            qvals = torch.stack(
                [quantum_filter(col, self.qweights[f]) for col in cols],
                dim=0
            )
            filter_outputs.append(qvals.view(batch_size, 16))

        conv_output = torch.stack(filter_outputs, dim=1)

        out = conv_output.reshape(batch_size, -1)   # (batch, 64)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn1(out)
        logits = self.fc2(out)
        return logits


@torch.no_grad()
def evaluate(model, X_tensor, y_tensor, criterion):
    model.eval()
    output = model(X_tensor)
    loss = criterion(output, y_tensor.unsqueeze(1))

    probs = torch.sigmoid(output)
    predicted = (probs >= 0.5).float().squeeze(1)
    correct = (predicted == y_tensor).sum().item()
    accuracy = correct / X_tensor.size(0) * 100.0
    return loss.item(), accuracy


model = QCNNClassifier()
model.apply(he_init_weights)

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=75, gamma=0.35)
criterion = nn.BCEWithLogitsLoss()

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

print(f"X_train_tensor shape = {X_tensor.shape}, y_train_tensor shape = {y_tensor.shape}")
print(f"X_val_tensor shape = {X_val_tensor.shape}, y_val_tensor shape = {y_val_tensor.shape}")

batch_size = 32
subset_size = X_tensor.shape[0]
num_epochs = 10
global_step = 0

best_loss = float("inf")
best_state = None

for epoch in range(num_epochs):
    model.train()
    indices = torch.randperm(subset_size)
    epoch_loss = 0.0

    for i in range(0, subset_size, batch_size):
        batch_indices = indices[i:i + batch_size]
        X_batch = X_tensor[batch_indices]
        y_batch = y_tensor[batch_indices]

        pred = model(X_batch)
        loss = criterion(pred, y_batch.unsqueeze(1))
        print(f"loss = {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        global_step += 1

    avg_loss = epoch_loss / ((subset_size + batch_size - 1) // batch_size)
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    val_loss, val_acc = evaluate(model, X_val_tensor, y_val_tensor, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

if best_state is not None:
    model.load_state_dict(best_state)

test_loss, test_acc = evaluate(model, X_test_tensor, y_test_tensor, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

torch.save(
    model.state_dict(),
    "quanvh_r5.pth"
)
