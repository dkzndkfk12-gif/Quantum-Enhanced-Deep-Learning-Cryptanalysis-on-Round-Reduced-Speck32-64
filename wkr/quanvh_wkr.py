import os
os.environ['OMP_NUM_THREADS'] = '1'

import copy
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import Speck1 as sp
from Speck1 import convert_to_binary

MODEL_PATH = 'quanvh_r5.pth'
OUTPUT_PATH = 'quanvh_wkr_r5.npz'
R_TRAIN = 5
DIFF = (0x0040, 0x0000)
TRIALS = 10
N_ITER = 10
KEY_BITS = 16
KEY_BATCH_SIZE = 32
SAMPLE_BATCH_SIZE = 256
SEED = None

if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

torch.set_num_threads(1)
DEVICE = torch.device('cpu')
MASK = 0xFFFF
ALPHA = 7
BETA = 2
NUM_QUBITS = 4
NUM_LAYERS = 4
NUM_FILTERS = 4
HIDDEN_DIM = 64
R_EVAL = R_TRAIN + 1


def rol(x, r):
    return ((x << r) & MASK) | (x >> (16 - r))


def ror(x, r):
    return (x >> r) | ((x << (16 - r)) & MASK)


def dec_one_round_vec(c0, c1, k):
    x = np.broadcast_to(np.asarray(c0, dtype=np.uint16), (len(k), TRIALS))
    y = np.broadcast_to(np.asarray(c1, dtype=np.uint16), (len(k), TRIALS))
    k = np.asarray(k, dtype=np.uint16)[:, None]
    y = ror(y ^ x, BETA).astype(np.uint16)
    x = rol(((x ^ k) - y) & MASK, ALPHA).astype(np.uint16)
    return x, y


dev = qml.device('lightning.qubit', wires=NUM_QUBITS)


@qml.qnode(dev, interface='torch', diff_method='adjoint')
def quantum_filter(inputs, weights):
    for i in range(NUM_QUBITS):
        qml.RX(inputs[i], wires=i)
    for layer in range(NUM_LAYERS):
        for i in range(NUM_QUBITS):
            qml.RX(weights[layer, i], wires=i)
        for i in range(NUM_QUBITS):
            qml.RZ(weights[layer, 4 + i], wires=i)
        qml.CNOT(wires=[3, 2])
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(0))


class QCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.qweights = nn.Parameter(0.05 * torch.randn(NUM_FILTERS, NUM_LAYERS, 8))
        self.fc1 = nn.Linear(NUM_FILTERS * 16, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        bsz = x.size(0)
        cols = (x * np.pi).transpose(1, 2).contiguous().view(-1, 4)
        outs = []
        for f in range(NUM_FILTERS):
            qvals = torch.stack([quantum_filter(col, self.qweights[f]) for col in cols], dim=0)
            outs.append(qvals.view(bsz, 16))
        z = torch.stack(outs, dim=1).reshape(bsz, -1)
        return self.fc2(self.bn1(self.relu(self.fc1(z))))


@torch.inference_mode()
def score_deltas(model, c0l, c0r, c1l, c1r, k_last, deltas):
    keys = np.uint16(k_last) ^ np.asarray(deltas, dtype=np.uint16)
    u0l, u0r = dec_one_round_vec(c0l, c0r, keys)
    u1l, u1r = dec_one_round_vec(c1l, c1r, keys)
    x = convert_to_binary((u0l.reshape(-1), u0r.reshape(-1), u1l.reshape(-1), u1r.reshape(-1)))
    x = torch.from_numpy(x).float().reshape(len(keys) * TRIALS, 4, 16)

    probs = np.empty(len(keys) * TRIALS, dtype=np.float32)
    for s in range(0, len(x), SAMPLE_BATCH_SIZE):
        e = min(s + SAMPLE_BATCH_SIZE, len(x))
        logits = model(x[s:e]).squeeze(1)
        probs[s:e] = torch.sigmoid(logits).cpu().numpy()

    probs = probs.reshape(len(keys), TRIALS)
    return probs.mean(axis=1), probs.std(axis=1)


model = QCNNClassifier().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

mu_accum = np.zeros(1 << KEY_BITS, dtype=np.float32)
sigma_accum = np.zeros(1 << KEY_BITS, dtype=np.float32)

for it in range(N_ITER):
    p0l = np.frombuffer(np.random.bytes(2 * TRIALS), dtype=np.uint16)
    p0r = np.frombuffer(np.random.bytes(2 * TRIALS), dtype=np.uint16)
    p1l = p0l ^ DIFF[0]
    p1r = p0r ^ DIFF[1]

    master_key = np.frombuffer(np.random.bytes(8), dtype=np.uint16)
    ks = sp.expand_key(master_key, R_EVAL)
    k_last = int(ks[-1])

    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    for s in range(0, 1 << KEY_BITS, KEY_BATCH_SIZE):
        e = min(s + KEY_BATCH_SIZE, 1 << KEY_BITS)
        deltas = np.arange(s, e, dtype=np.uint16)
        mu, sigma = score_deltas(model, c0l, c0r, c1l, c1r, k_last, deltas)
        mu_accum[s:e] += mu
        sigma_accum[s:e] += sigma
        if s % 4096 == 0:
            print(f'iter={it+1}/{N_ITER}  delta={s:5d}..{e-1:5d}  mu0={mu[0]:.4f}  sigma0={sigma[0]:.4f}')

mu = mu_accum / N_ITER
sigma = sigma_accum / N_ITER
np.savez(OUTPUT_PATH, mu=mu, sigma=sigma)
print(f"saved to '{OUTPUT_PATH}'")
