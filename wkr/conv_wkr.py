import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.nn as nn
import Speck1 as sp
from Speck1 import convert_to_binary

MODEL_PATH = 'classical_conv.pth'
OUTPUT_PATH = 'conv_wkr_r5.npz'
R_TRAIN = 5
DIFF = (0x0040, 0x0000)
TRIALS = 10
N_ITER = 10
KEY_BITS = 16
KEY_BATCH_SIZE = 256
SAMPLE_BATCH_SIZE = 4096
SEED = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

torch.set_num_threads(1)
DEVICE = torch.device(DEVICE)
MASK = 0xFFFF
ALPHA = 7
BETA = 2
HIDDEN_DIM = 64
R_EVAL = R_TRAIN + 1


def rol(x, r):
    return ((x << r) & MASK) | (x >> (16 - r))


def ror(x, r):
    return (x >> r) | ((x << (16 - r)) & MASK)


def dec_one_round_vec(c0, c1, keys):
    x = np.broadcast_to(np.asarray(c0, dtype=np.uint16), (len(keys), TRIALS))
    y = np.broadcast_to(np.asarray(c1, dtype=np.uint16), (len(keys), TRIALS))
    k = np.asarray(keys, dtype=np.uint16)[:, None]
    y = ror(y ^ x, BETA).astype(np.uint16)
    x = rol(((x ^ k) - y) & MASK, ALPHA).astype(np.uint16)
    return x, y


class ConvStrict(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=(4, 1), bias=True)
        self.bn_conv = nn.BatchNorm2d(4)
        self.relu_conv = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 4, 16)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bn_conv(x)
        x = self.relu_conv(x)
        x = x.squeeze(2)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return self.fc2(x)


@torch.inference_mode()
def score_deltas(model, c0l, c0r, c1l, c1r, k_last, deltas):
    keys = np.uint16(k_last) ^ np.asarray(deltas, dtype=np.uint16)
    u0l, u0r = dec_one_round_vec(c0l, c0r, keys)
    u1l, u1r = dec_one_round_vec(c1l, c1r, keys)
    x = convert_to_binary((u0l.reshape(-1), u0r.reshape(-1), u1l.reshape(-1), u1r.reshape(-1)))
    x = torch.from_numpy(x).float().reshape(len(keys) * TRIALS, 4, 16).to(DEVICE)

    probs = np.empty(len(keys) * TRIALS, dtype=np.float32)
    for s in range(0, len(x), SAMPLE_BATCH_SIZE):
        e = min(s + SAMPLE_BATCH_SIZE, len(x))
        logits = model(x[s:e]).squeeze(1)
        probs[s:e] = torch.sigmoid(logits).cpu().numpy()

    probs = probs.reshape(len(keys), TRIALS)
    return probs.mean(axis=1), probs.std(axis=1)


model = ConvStrict().to(DEVICE)
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
