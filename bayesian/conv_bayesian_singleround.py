import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import math
from os import urandom
from time import time

import numpy as np
import torch
import torch.nn as nn
import Speck1 as sp

# =========================
# Edit only this block
# =========================
MODEL_PATH = 'conv_strict_r6.pth'
WKR_PATH = 'conv_wkr_r6.npz'
OUTPUT_PATH = 'conv_bayes_one_round.npz'

DEVICE = 'cpu'
SAMPLE_BATCH_SIZE = 4096
LOW_BITS = 12
NUM_CAND = 32
NUM_ITER_STAGE = 5
VERIFY_BREADTH = 64
EPS = 1e-6

ATTACK_TRIALS = 5
NR = 9
NUM_STRUCTURES = 100
OUTER_ITERS = 200
CUTOFF = 5.0
DIFF_CHALLENGE = (0x0211, 0x0A04)
TARGET_DIFF = (0x0040, 0x0000)
NEUTRAL_BITS = [20, 21, 22, 14, 15, 23]
KEYSCHEDULE = 'real'
SEED = None
# =========================

if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

torch.set_num_threads(1)
DEVICE = torch.device(DEVICE)
WORD_SIZE = sp.WORD_SIZE()
MASK = (1 << WORD_SIZE) - 1
ALPHA = 7
BETA = 2
HIDDEN_DIM = 64
LOW_KEYSPACE = 1 << LOW_BITS
HIGH_BITS = WORD_SIZE - LOW_BITS


def rol(x, r):
    x = np.asarray(x, dtype=np.uint16)
    return ((x << r) & MASK) | (x >> (WORD_SIZE - r))


def ror(x, r):
    x = np.asarray(x, dtype=np.uint16)
    return (x >> r) | ((x << (WORD_SIZE - r)) & MASK)


def dec_one_round_vec(c0, c1, keys):
    x = np.broadcast_to(np.asarray(c0, dtype=np.uint16), (len(keys), len(c0)))
    y = np.broadcast_to(np.asarray(c1, dtype=np.uint16), (len(keys), len(c1)))
    k = np.asarray(keys, dtype=np.uint16)[:, None]
    y = ror(y ^ x, BETA).astype(np.uint16)
    x = rol(((x ^ k) - y) & MASK, ALPHA).astype(np.uint16)
    return x, y


def convert_to_binary(words):
    if hasattr(sp, 'convert_to_binary'):
        return sp.convert_to_binary(words)
    words = [np.asarray(w, dtype=np.uint16).reshape(-1) for w in words]
    X = np.zeros((4 * WORD_SIZE, len(words[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE):
        wi = i // WORD_SIZE
        off = WORD_SIZE - 1 - (i % WORD_SIZE)
        X[i] = (words[wi] >> off) & 1
    return X.T


def load_profile(path):
    z = np.load(path)
    mu = z['mu'].astype(np.float32)
    sigma = np.maximum(z['sigma'].astype(np.float32), EPS)
    return mu, (1.0 / sigma).astype(np.float32)


def hw(v):
    v = np.asarray(v, dtype=np.uint16)
    out = np.zeros(v.shape, dtype=np.uint8)
    for i in range(WORD_SIZE):
        out += ((v >> i) & 1).astype(np.uint8)
    return out


low_weight = np.arange(1 << WORD_SIZE, dtype=np.uint16)
low_weight = low_weight[hw(low_weight) <= 2]


def make_structure(pt0, pt1, diff=DIFF_CHALLENGE, neutral_bits=NEUTRAL_BITS):
    p0 = np.copy(pt0).reshape(-1, 1)
    p1 = np.copy(pt1).reshape(-1, 1)
    for bit in neutral_bits:
        d = 1 << bit
        d0, d1 = d >> 16, d & 0xFFFF
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
    return p0, p1, p0 ^ diff[0], p1 ^ diff[1]


def gen_plain(n):
    return (np.frombuffer(urandom(2 * n), dtype=np.uint16),
            np.frombuffer(urandom(2 * n), dtype=np.uint16))


def gen_key(nr):
    return sp.expand_key(np.frombuffer(urandom(8), dtype=np.uint16), nr)


def gen_challenge(num_structures, nr=NR, diff=DIFF_CHALLENGE, neutral_bits=NEUTRAL_BITS, keyschedule=KEYSCHEDULE):
    pt0, pt1 = gen_plain(num_structures)
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), 0)
    pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), 0)
    key = gen_key(nr)
    if keyschedule == 'free':
        key = np.frombuffer(urandom(2 * nr), dtype=np.uint16)
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), key)
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), key)
    return [ct0a, ct1a, ct0b, ct1b], key


def find_good(cts, key, nr=3, target_diff=TARGET_DIFF):
    pt0a, pt1a = sp.decrypt((cts[0], cts[1]), key[nr:])
    pt0b, pt1b = sp.decrypt((cts[2], cts[3]), key[nr:])
    d = (pt0a ^ pt0b == target_diff[0]) & (pt1a ^ pt1b == target_diff[1])
    return np.sum(d, axis=1)


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
        return self.fc2(x).squeeze(1)


def load_model(path):
    model = ConvStrict().to(DEVICE)
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


def bayes_rank(cand, emp_mean, mu, inv_sigma):
    base = np.arange(LOW_KEYSPACE, dtype=np.uint16)[:, None]
    idx = base ^ np.asarray(cand, dtype=np.uint16)[None, :]
    z = (np.asarray(emp_mean, dtype=np.float32)[None, :] - mu[idx]) * inv_sigma[idx]
    return np.linalg.norm(z, axis=1)


def bayes_keysearch_one_round(struct, model, mu, inv_sigma, num_cand=NUM_CAND, num_iter=NUM_ITER_STAGE):
    n = len(struct[0])
    keys = np.random.choice(LOW_KEYSPACE, num_cand, replace=False).astype(np.uint16)
    best_key, best_val = 0, -np.inf

    for _ in range(num_iter):
        u0l, u0r = dec_one_round_vec(struct[0], struct[1], keys)
        u1l, u1r = dec_one_round_vec(struct[2], struct[3], keys)
        X = convert_to_binary((u0l.reshape(-1), u0r.reshape(-1), u1l.reshape(-1), u1r.reshape(-1)))
        probs = predict_proba(model, X).reshape(len(keys), n)
        means = probs.mean(axis=1)
        vals = np.log2(probs / (1.0 - probs)).sum(axis=1)

        m = int(np.argmax(vals))
        if vals[m] > best_val:
            best_val = float(vals[m])
            best_key = int(keys[m])

        base_keys = np.argpartition(bayes_rank(keys, means, mu, inv_sigma), num_cand)[:num_cand].astype(np.uint16)
        if HIGH_BITS > 0:
            hi = np.random.randint(0, 1 << HIGH_BITS, size=num_cand, dtype=np.uint16) << LOW_BITS
            keys = base_keys ^ hi
        else:
            keys = base_keys
    return best_key, best_val


def hamming_verify_one_round(struct, k_guess, model, use_pairs=VERIFY_BREADTH):
    use_pairs = min(use_pairs, len(struct[0]))
    cand = np.uint16(k_guess) ^ low_weight
    u0l, u0r = dec_one_round_vec(struct[0][:use_pairs], struct[1][:use_pairs], cand)
    u1l, u1r = dec_one_round_vec(struct[2][:use_pairs], struct[3][:use_pairs], cand)
    X = convert_to_binary((u0l.reshape(-1), u0r.reshape(-1), u1l.reshape(-1), u1r.reshape(-1)))
    probs = predict_proba(model, X).reshape(len(cand), use_pairs)
    vals = np.log2(probs / (1.0 - probs)).mean(axis=1) * len(struct[0])
    idx = int(np.argmax(vals))
    return int(cand[idx]), float(vals[idx])


def attack_once(model, mu, inv_sigma, n_struct=NUM_STRUCTURES, iter_struct=OUTER_ITERS, cutoff=CUTOFF, verbose=False):
    cts, key_sched = gen_challenge(n_struct, NR)
    real_k = int(key_sched[-1])
    n = len(cts[0])
    alpha = math.sqrt(n)
    local_best = np.full(n, -1e9, dtype=np.float32)
    visits = np.full(n, 1e-3, dtype=np.float32)
    best_k, best_v, best_idx = 0, -np.inf, 0

    for j in range(iter_struct):
        priority = local_best + alpha * np.sqrt(np.log2(j + 2) / visits)
        idx = int(np.argmax(priority))
        visits[idx] += 1.0
        k, v = bayes_keysearch_one_round([cts[0][idx], cts[1][idx], cts[2][idx], cts[3][idx]], model, mu, inv_sigma)
        if v > local_best[idx]:
            local_best[idx] = v
        if v > best_v:
            best_k, best_v, best_idx = k, v, idx
        if best_v > cutoff:
            break

    if VERIFY_BREADTH > 0:
        best_k, best_v = hamming_verify_one_round([cts[0][best_idx], cts[1][best_idx], cts[2][best_idx], cts[3][best_idx]], best_k, model)
    if verbose:
        print(f'real=0x{real_k:04X} guess=0x{best_k:04X} diff=0x{real_k ^ best_k:04X}')
    return real_k, best_k, int(np.max(find_good(cts, key_sched)))


def main():
    if not sp.check_testvector():
        raise RuntimeError('Speck1 test vector mismatch')

    model = load_model(MODEL_PATH)
    mu, inv_sigma = load_profile(WKR_PATH)

    real = np.zeros(ATTACK_TRIALS, dtype=np.uint16)
    guess = np.zeros(ATTACK_TRIALS, dtype=np.uint16)
    diff = np.zeros(ATTACK_TRIALS, dtype=np.uint16)
    good = np.zeros(ATTACK_TRIALS, dtype=np.uint16)
    t0 = time()

    for i in range(ATTACK_TRIALS):
        print(f'Trial {i + 1}/{ATTACK_TRIALS}')
        rk, gk, gd = attack_once(model, mu, inv_sigma, verbose=True)
        real[i], guess[i], diff[i], good[i] = rk, gk, np.uint16(rk ^ gk), gd

    dt = (time() - t0) / ATTACK_TRIALS
    succ = int(np.sum(diff == 0))
    np.savez(OUTPUT_PATH, real=real, guess=guess, diff=diff, good=good)
    print(f'Success {succ}/{ATTACK_TRIALS}')
    print(f'Avg time {dt:.2f}s')
    print(f"saved to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    main()
