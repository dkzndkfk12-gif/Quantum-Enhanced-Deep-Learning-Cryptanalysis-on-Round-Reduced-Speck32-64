import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import math
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import Speck1 as sp
from os import urandom
from time import time

# =========================
# Edit only this block
# =========================
MODEL_PATH = 'quanvh_r5.pth'
WKR_PATH = 'qcnn_wkr_r5.npz'
OUTPUT_PATH = 'quanvh_bayes_one_round_r5_to_r6.npz'

DEVICE = 'cpu'
SAMPLE_BATCH_SIZE = 256
EPS = 1e-6

R_TRAIN = 5
DIFF = (0x0040, 0x0000)
NUM_PAIRS = 10
TRIALS = 5
NUM_CAND = 32
NUM_ITER = 5
VERIFY_BREADTH = 10
SEED = None
# =========================

if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

torch.set_num_threads(1)
DEVICE = torch.device(DEVICE)
WORD_SIZE = sp.WORD_SIZE()
NUM_QUBITS = 4
NUM_LAYERS = 4
NUM_FILTERS = 4
HIDDEN_DIM = 64


def hw(v):
    v = np.asarray(v, dtype=np.uint16)
    out = np.zeros(v.shape, dtype=np.uint8)
    for i in range(WORD_SIZE):
        out += ((v >> i) & 1).astype(np.uint8)
    return out


low_weight = np.arange(1 << WORD_SIZE, dtype=np.uint16)
low_weight = low_weight[hw(low_weight) <= 2]


def load_profile(path):
    z = np.load(path)
    mu = z['mu'].astype(np.float32)
    sigma = np.maximum(z['sigma'].astype(np.float32), EPS)
    return mu, (1.0 / sigma).astype(np.float32)


def gen_key(nr):
    return sp.expand_key(np.frombuffer(urandom(8), dtype=np.uint16), nr)


def gen_plain_pairs(n, diff=DIFF):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r


def gen_cipher_pairs(n, nr=R_TRAIN + 1, diff=DIFF):
    p0l, p0r, p1l, p1r = gen_plain_pairs(n, diff=diff)
    ks = gen_key(nr)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)
    return (c0l, c0r, c1l, c1r), ks


dev = qml.device('lightning.qubit', wires=NUM_QUBITS)


@qml.qnode(dev, interface='torch', diff_method='adjoint')
def quantum_filter(inputs, weights):
    for i in range(NUM_QUBITS):
        qml.RX(inputs[i], wires=i)
    for l in range(NUM_LAYERS):
        for i in range(NUM_QUBITS):
            qml.RX(weights[l, i], wires=i)
        for i in range(NUM_QUBITS):
            qml.RZ(weights[l, 4 + i], wires=i)
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
        b = x.size(0)
        cols = (x * math.pi).transpose(1, 2).contiguous().view(-1, 4)
        qs = []
        for f in range(NUM_FILTERS):
            q = torch.stack([quantum_filter(col, self.qweights[f]) for col in cols], dim=0)
            qs.append(q.view(b, 16))
        x = torch.stack(qs, dim=1).reshape(b, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        return self.fc2(x).squeeze(1)


def load_model(path):
    model = QCNNClassifier().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def predict_proba(model, X_bits):
    if X_bits.ndim == 2:
        X_bits = X_bits.reshape(-1, 4, 16)
    X_bits = X_bits.astype(np.float32, copy=False)
    out = np.empty(len(X_bits), dtype=np.float32)
    for s in range(0, len(X_bits), SAMPLE_BATCH_SIZE):
        xb = torch.from_numpy(X_bits[s:s + SAMPLE_BATCH_SIZE]).to(DEVICE)
        out[s:s + SAMPLE_BATCH_SIZE] = torch.sigmoid(model(xb)).cpu().numpy()
    return np.clip(out, EPS, 1.0 - EPS)


def score_candidates(c0l, c0r, c1l, c1r, cand_keys, model):
    cand_keys = np.asarray(cand_keys, dtype=np.uint16)
    blocks = []
    for k in cand_keys:
        u0l, u0r = sp.dec_one_round((c0l, c0r), int(k))
        u1l, u1r = sp.dec_one_round((c1l, c1r), int(k))
        X = sp.convert_to_binary((u0l, u0r, u1l, u1r))
        blocks.append(X)
    X_all = np.concatenate(blocks, axis=0)
    probs = predict_proba(model, X_all).reshape(len(cand_keys), len(c0l))
    means = probs.mean(axis=1)
    vals = np.log2(probs / (1.0 - probs)).sum(axis=1)
    return means.astype(np.float32), vals.astype(np.float32)


def bayes_rank_all(cand_keys, emp_mean, mu, inv_sigma):
    all_keys = np.arange(1 << WORD_SIZE, dtype=np.uint16)
    idx = np.bitwise_xor(all_keys[:, None], np.asarray(cand_keys, dtype=np.uint16)[None, :])
    z = (np.asarray(emp_mean, dtype=np.float32)[None, :] - mu[idx]) * inv_sigma[idx]
    return np.linalg.norm(z, axis=1)


def bayesian_keysearch_one_round(cipher_pairs, model, mu, inv_sigma, num_cand=NUM_CAND, num_iter=NUM_ITER):
    c0l, c0r, c1l, c1r = cipher_pairs
    cand = np.random.choice(1 << WORD_SIZE, size=num_cand, replace=False).astype(np.uint16)
    best_key = 0
    best_val = -np.inf

    for _ in range(num_iter):
        means, vals = score_candidates(c0l, c0r, c1l, c1r, cand, model)
        m = int(np.argmax(vals))
        if vals[m] > best_val:
            best_val = float(vals[m])
            best_key = int(cand[m])
        scores = bayes_rank_all(cand, means, mu, inv_sigma)
        cand = np.argpartition(scores, num_cand)[:num_cand].astype(np.uint16)

    return best_key, best_val


def hamming_verify_one_round(cipher_pairs, k_guess, model, use_pairs=VERIFY_BREADTH):
    c0l, c0r, c1l, c1r = cipher_pairs
    use_pairs = min(use_pairs, len(c0l))
    cand = np.uint16(k_guess) ^ low_weight
    means, vals = score_candidates(c0l[:use_pairs], c0r[:use_pairs], c1l[:use_pairs], c1r[:use_pairs], cand, model)
    idx = int(np.argmax(vals))
    return int(cand[idx]), float(vals[idx])


def attack_once(model, mu, inv_sigma, num_pairs=NUM_PAIRS, verbose=False):
    cipher_pairs, round_keys = gen_cipher_pairs(num_pairs, nr=R_TRAIN + 1, diff=DIFF)
    real_k = int(round_keys[-1])
    guess_k, score0 = bayesian_keysearch_one_round(cipher_pairs, model, mu, inv_sigma)
    if VERIFY_BREADTH > 0:
        guess_k, score1 = hamming_verify_one_round(cipher_pairs, guess_k, model)
    else:
        score1 = score0
    if verbose:
        print(f'real=0x{real_k:04X} guess=0x{guess_k:04X} diff=0x{real_k ^ guess_k:04X} score={score1:.4f}')
    return real_k, guess_k, score0, score1


def main():
    if not sp.check_testvector():
        raise RuntimeError('Speck1 test vector mismatch')

    model = load_model(MODEL_PATH)
    mu, inv_sigma = load_profile(WKR_PATH)

    real = np.zeros(TRIALS, dtype=np.uint16)
    guess = np.zeros(TRIALS, dtype=np.uint16)
    diff = np.zeros(TRIALS, dtype=np.uint16)
    score_before = np.zeros(TRIALS, dtype=np.float32)
    score_after = np.zeros(TRIALS, dtype=np.float32)

    t0 = time()
    for i in range(TRIALS):
        print(f'Trial {i + 1}/{TRIALS}')
        rk, gk, s0, s1 = attack_once(model, mu, inv_sigma, verbose=True)
        real[i] = rk
        guess[i] = gk
        diff[i] = np.uint16(rk ^ gk)
        score_before[i] = s0
        score_after[i] = s1

    dt = (time() - t0) / TRIALS
    succ = int(np.sum(diff == 0))
    np.savez(OUTPUT_PATH, real=real, guess=guess, diff=diff,
             score_before_verify=score_before, score_after_verify=score_after)
    print(f'Success {succ}/{TRIALS}')
    print(f'Avg time {dt:.2f}s')
    print(f"saved to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    main()
