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

MODEL_PATH_R5 = 'qcnn_Speck_4by16_4qubit_circuit2_4layer_4filters_gamma035_second_260406_r5.pth'
MODEL_PATH_R6 = 'qcnn_Speck_4by16_4qubit_circuit2_4layer_4filters_gamma035_second_260406_r6.pth'
WKR_PATH_R5 = 'qcnn_wkr_r5.npz'
WKR_PATH_R6 = 'qcnn_wkr_r6.npz'
OUTPUT_PREFIX = 'qcnn_bayes_r10'

DEVICE = 'cpu'
SAMPLE_BATCH_SIZE = 256
LOW_BITS = 12
NUM_CAND = 32
NUM_ITER_STAGE = 1
VERIFY_BREADTH = 64
EPS = 1e-6

N_ATTACKS = 100
NR = 10
NUM_STRUCTURES = 100
OUTER_ITERS = 3
CUTOFF1 = 40.0
CUTOFF2 = 10.0
DIFF_CHALLENGE = (0x0211, 0x0A04)
TARGET_DIFF = (0x0040, 0x0000)
NEUTRAL_BITS = [20, 21, 22, 14, 15, 23]
KEYSCHEDULE = 'real'
SEED = None

if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

torch.set_num_threads(1)
DEVICE = torch.device(DEVICE)
WORD_SIZE = sp.WORD_SIZE()
MASK = (1 << WORD_SIZE) - 1
ALPHA = 7
BETA = 2
NUM_QUBITS = 4
NUM_LAYERS = 4
NUM_FILTERS = 4
HIDDEN_DIM = 64
HIGH_BITS = WORD_SIZE - LOW_BITS
LOW_KEYSPACE = 1 << LOW_BITS


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


def dec_one_round_single(c0, c1, key):
    return sp.dec_one_round((np.asarray(c0, dtype=np.uint16), np.asarray(c1, dtype=np.uint16)), np.uint16(key))


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


def make_structure(pt0, pt1, diff=DIFF_CHALLENGE, neutral_bits=NEUTRAL_BITS):
    p0 = np.copy(pt0).reshape(-1, 1)
    p1 = np.copy(pt1).reshape(-1, 1)
    for bit in neutral_bits:
        d = 1 << bit
        d0, d1 = d >> 16, d & 0xFFFF
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
    return p0, p1, p0 ^ diff[0], p1 ^ diff[1]


def gen_key(nr):
    return sp.expand_key(np.frombuffer(urandom(8), dtype=np.uint16), nr)


def gen_plain(n):
    return (np.frombuffer(urandom(2 * n), dtype=np.uint16),
            np.frombuffer(urandom(2 * n), dtype=np.uint16))


def gen_challenge(num_structures, nr, diff=DIFF_CHALLENGE, neutral_bits=NEUTRAL_BITS, keyschedule=KEYSCHEDULE):
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


def verifier_search(cts, best_guess, model, use_n=VERIFY_BREADTH):
    use_n = min(use_n, len(cts[0]))
    k1 = np.uint16(best_guess[0]) ^ low_weight
    k2 = np.uint16(best_guess[1]) ^ low_weight
    n = len(low_weight)
    keys1 = np.repeat(k1, n)
    keys2 = np.tile(k2, n)

    ct0a = np.tile(cts[0][:use_n], n * n)
    ct1a = np.tile(cts[1][:use_n], n * n)
    ct0b = np.tile(cts[2][:use_n], n * n)
    ct1b = np.tile(cts[3][:use_n], n * n)

    p0a, p1a = dec_one_round_vec(ct0a, ct1a, keys1)
    p0b, p1b = dec_one_round_vec(ct0b, ct1b, keys1)
    p0a, p1a = dec_one_round_vec(p0a.reshape(-1), p1a.reshape(-1), np.repeat(keys2, use_n))
    p0b, p1b = dec_one_round_vec(p0b.reshape(-1), p1b.reshape(-1), np.repeat(keys2, use_n))

    X = convert_to_binary((p0a.reshape(-1), p1a.reshape(-1), p0b.reshape(-1), p1b.reshape(-1)))
    probs = predict_proba(model, X).reshape(n * n, use_n)
    vals = np.log2(probs / (1.0 - probs)).mean(axis=1) * len(cts[0])
    idx = int(np.argmax(vals))
    return int(keys1[idx]), int(keys2[idx]), float(vals[idx])


def bayesian_rank_kr(cand, emp_mean, mu, inv_sigma):
    base = np.arange(LOW_KEYSPACE, dtype=np.uint16)[:, None]
    tmp = base ^ np.asarray(cand, dtype=np.uint16)[None, :]
    z = (np.asarray(emp_mean, dtype=np.float32)[None, :] - mu[tmp]) * inv_sigma[tmp]
    return np.linalg.norm(z, axis=1)


def bayesian_key_recovery(cts, model, mu, inv_sigma, num_cand=NUM_CAND, num_iter=NUM_ITER_STAGE, seed=None):
    n = len(cts[0])
    keys = np.array(seed, dtype=np.uint16) if seed is not None else np.random.choice(LOW_KEYSPACE, num_cand, replace=False).astype(np.uint16)
    all_keys = np.zeros(num_cand * num_iter, dtype=np.uint16)
    all_vals = np.full(num_cand * num_iter, -np.inf, dtype=np.float32)

    for i in range(num_iter):
        u0l, u0r = dec_one_round_vec(cts[0], cts[1], keys)
        u1l, u1r = dec_one_round_vec(cts[2], cts[3], keys)
        X = convert_to_binary((u0l.reshape(-1), u0r.reshape(-1), u1l.reshape(-1), u1r.reshape(-1)))
        probs = predict_proba(model, X).reshape(len(keys), n)
        means = probs.mean(axis=1)
        vals = np.log2(probs / (1.0 - probs)).sum(axis=1)

        all_keys[i * num_cand:(i + 1) * num_cand] = keys
        all_vals[i * num_cand:(i + 1) * num_cand] = vals

        base_keys = np.argpartition(bayesian_rank_kr(keys, means, mu, inv_sigma), num_cand)[:num_cand].astype(np.uint16)
        if HIGH_BITS > 0:
            hi = np.random.randint(0, 1 << HIGH_BITS, size=num_cand, dtype=np.uint16) << LOW_BITS
            keys = base_keys ^ hi
        else:
            keys = base_keys
    return all_keys, all_vals


def test_bayes(cts, model_main, model_help, mu_main, inv_main, mu_help, inv_help,
               it=OUTER_ITERS, cutoff1=CUTOFF1, cutoff2=CUTOFF2, verify_breadth=VERIFY_BREADTH):
    n_struct = len(cts[0])
    if verify_breadth is None:
        verify_breadth = len(cts[0][0])

    alpha = math.sqrt(n_struct)
    local_best = np.full(n_struct, -1e9, dtype=np.float32)
    num_visits = np.full(n_struct, 1e-3, dtype=np.float32)
    best_val = -np.inf
    best_key = (0, 0)
    best_pod = 0
    steps_used = it

    for j in range(it):
        priority = local_best + alpha * np.sqrt(np.log2(j + 2) / num_visits)
        pod = int(np.argmax(priority))
        num_visits[pod] += 1.0
        steps_used = j + 1

        keys1, vals1 = bayesian_key_recovery([cts[0][pod], cts[1][pod], cts[2][pod], cts[3][pod]], model_main, mu_main, inv_main)
        v1max = float(np.max(vals1))
        local_best[pod] = max(local_best[pod], v1max)

        if v1max > cutoff1:
            for idx in np.where(vals1 > cutoff1)[0]:
                k1 = int(keys1[idx])
                c0a, c1a = dec_one_round_single(cts[0][pod], cts[1][pod], k1)
                c0b, c1b = dec_one_round_single(cts[2][pod], cts[3][pod], k1)
                keys2, vals2 = bayesian_key_recovery([c0a, c1a, c0b, c1b], model_help, mu_help, inv_help)
                v2max = float(np.max(vals2))
                if v2max > best_val:
                    best_val = v2max
                    best_key = (k1, int(keys2[np.argmax(vals2)]))
                    best_pod = pod

        if best_val > cutoff2:
            break

    if verify_breadth and best_val > -np.inf:
        improved = True
        while improved:
            k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]],
                                          best_key, model_help, use_n=verify_breadth)
            improved = val > best_val
            if improved:
                best_key = (k1, k2)
                best_val = val
    return best_key, steps_used


def run_attacks(n_attacks=N_ATTACKS, nr=NR, num_structures=NUM_STRUCTURES, verify_breadth=VERIFY_BREADTH):
    if not sp.check_testvector():
        raise RuntimeError('Speck1 test vector mismatch')

    model_r5 = load_model(MODEL_PATH_R5)
    model_r6 = load_model(MODEL_PATH_R6)
    mu5, inv5 = load_profile(WKR_PATH_R5)
    mu6, inv6 = load_profile(WKR_PATH_R6)

    arr1 = np.zeros(n_attacks, dtype=np.uint16)
    arr2 = np.zeros(n_attacks, dtype=np.uint16)
    good = np.zeros(n_attacks, dtype=np.uint16)
    data = 0
    t0 = time()

    for i in range(n_attacks):
        print(f'Test {i + 1}/{n_attacks}')
        cts, key = gen_challenge(num_structures, nr)
        good[i] = np.max(find_good(cts, key))
        guess, used = test_bayes(cts, model_r6, model_r5, mu6, inv6, mu5, inv5,
                                 it=OUTER_ITERS, cutoff1=CUTOFF1, cutoff2=CUTOFF2,
                                 verify_breadth=verify_breadth)
        arr1[i] = np.uint16(guess[0] ^ key[nr - 1])
        arr2[i] = np.uint16(guess[1] ^ key[nr - 2])
        data += 2 * (2 ** len(NEUTRAL_BITS)) * min(num_structures, used)
        print('Difference:', hex(int(arr1[i])), hex(int(arr2[i])))

    t1 = time()
    np.savez(f'{OUTPUT_PREFIX}.npz', arr1=arr1, arr2=arr2, good=good)
    print('Done.')
    print('Last-round diffs:', [hex(int(x)) for x in arr1])
    print('Second-last diffs:', [hex(int(x)) for x in arr2])
    print('Wall time per attack:', (t1 - t0) / n_attacks)
    print('Data blocks used (average, log2):', math.log2(data) - math.log2(n_attacks))
    print(f"saved to '{OUTPUT_PREFIX}.npz'")


if __name__ == '__main__':
    run_attacks()
