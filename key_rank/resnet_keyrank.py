import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import math
import numpy as np
import torch
import torch.nn as nn

# =========================
# Edit only this block
# =========================
MODEL_PATH = 'classical_resnet.pth'
OUTPUT_PATH = 'resnet_key_ranking_results.npz'
R_TRAIN = 5
DIFF = (0x0040, 0x0000)
NUM_PAIRS = 10
N_ITER = 1
KEY_BATCH_SIZE = 64
SAMPLE_BATCH_SIZE = 4096
SEED = 1234
DEVICE = 'cpu'
# =========================

WORD_SIZE = 16
MASK = (1 << WORD_SIZE) - 1
ALPHA = 7
BETA = 2
NUM_FILTERS = 4
HIDDEN_DIM = 64
DEPTH = 10


def _try_import_speck1():
    try:
        import Speck1 as sp
        need = ['expand_key', 'encrypt', 'dec_one_round', 'convert_to_binary']
        if all(hasattr(sp, x) for x in need):
            return sp
    except Exception:
        pass
    return None


SP = _try_import_speck1()


def rol(x, r):
    x = np.asarray(x, dtype=np.uint16)
    return ((x << r) & MASK) | (x >> (WORD_SIZE - r))


def ror(x, r):
    x = np.asarray(x, dtype=np.uint16)
    return (x >> r) | ((x << (WORD_SIZE - r)) & MASK)


def enc_one_round(p, k):
    x, y = p
    x = ror(x, ALPHA)
    x = (x + y) & MASK
    x = x ^ np.asarray(k, dtype=np.uint16)
    y = rol(y, BETA)
    y = y ^ x
    return x.astype(np.uint16), y.astype(np.uint16)


def dec_one_round_local(c, k):
    x, y = c
    y = y ^ x
    y = ror(y, BETA)
    x = x ^ np.asarray(k, dtype=np.uint16)
    x = (x - y) & MASK
    x = rol(x, ALPHA)
    return x.astype(np.uint16), y.astype(np.uint16)


def expand_key_local(k, rounds):
    k = [int(v) & MASK for v in np.asarray(k, dtype=np.uint16)]
    ks = [0] * rounds
    ks[0] = k[-1]
    l = list(reversed(k[:-1]))
    for i in range(rounds - 1):
        j = i % len(l)
        l[j], ks[i + 1] = enc_one_round((np.uint16(l[j]), np.uint16(ks[i])), np.uint16(i))
        l[j], ks[i + 1] = int(l[j]), int(ks[i + 1])
    return np.asarray(ks, dtype=np.uint16)


def encrypt_local(p, ks):
    x = np.asarray(p[0], dtype=np.uint16)
    y = np.asarray(p[1], dtype=np.uint16)
    for k in np.asarray(ks, dtype=np.uint16):
        x, y = enc_one_round((x, y), k)
    return x, y


def convert_to_binary_local(words):
    words = [np.asarray(w, dtype=np.uint16).reshape(-1) for w in words]
    n = words[0].shape[0]
    X = np.zeros((4 * WORD_SIZE, n), dtype=np.uint8)
    for i in range(4 * WORD_SIZE):
        wi = i // WORD_SIZE
        off = WORD_SIZE - 1 - (i % WORD_SIZE)
        X[i] = (words[wi] >> off) & 1
    return X.T


def expand_key(k, rounds):
    return SP.expand_key(k, rounds) if SP else expand_key_local(k, rounds)


def encrypt(p, ks):
    return SP.encrypt(p, ks) if SP else encrypt_local(p, ks)


def dec_one_round(c, k):
    return SP.dec_one_round(c, k) if SP else dec_one_round_local(c, k)


def convert_to_binary(words):
    return SP.convert_to_binary(words) if SP else convert_to_binary_local(words)


class GohrStyleResidualBlock(nn.Module):
    def __init__(self, channels=NUM_FILTERS, ks=3):
        super().__init__()
        pad = ks // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=ks, padding=pad, bias=True)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=ks, padding=pad, bias=True)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out + shortcut


class ResNetDistinguisher(nn.Module):
    def __init__(self, num_filters=NUM_FILTERS, hidden_dim=HIDDEN_DIM, depth=DEPTH):
        super().__init__()
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(4, 1), bias=True)
        self.bn_conv = nn.BatchNorm2d(num_filters)
        self.relu_conv = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([GohrStyleResidualBlock(num_filters, 3) for _ in range(depth)])
        self.fc1 = nn.Linear(num_filters * WORD_SIZE, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 4, 16)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bn_conv(x)
        x = self.relu_conv(x)
        x = x.squeeze(2)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return self.fc2(x).squeeze(1)


def load_model(path, device):
    model = ResNetDistinguisher().to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def score_keys(model, c0l, c0r, c1l, c1r, keys, sample_batch_size, device):
    t = len(c0l)
    k = len(keys)
    keys2 = keys[:, None].astype(np.uint16)

    u0l, u0r = dec_one_round((np.broadcast_to(c0l, (k, t)), np.broadcast_to(c0r, (k, t))), keys2)
    u1l, u1r = dec_one_round((np.broadcast_to(c1l, (k, t)), np.broadcast_to(c1r, (k, t))), keys2)

    X = convert_to_binary((u0l.reshape(-1), u0r.reshape(-1), u1l.reshape(-1), u1r.reshape(-1)))
    X = X.reshape(k * t, 4, 16).astype(np.float32)

    logits = []
    for s in range(0, len(X), sample_batch_size):
        xb = torch.from_numpy(X[s:s + sample_batch_size]).to(device)
        logits.append(model(xb).cpu().numpy())
    logits = np.concatenate(logits).reshape(k, t)
    return logits.sum(axis=1) / math.log(2.0)


def main():
    rng = np.random.default_rng(SEED)
    model = load_model(MODEL_PATH, DEVICE)

    master_key = rng.integers(0, 1 << WORD_SIZE, size=4, dtype=np.uint16)
    round_keys = expand_key(master_key, R_TRAIN + 1)
    k_last = int(round_keys[-1])

    score_accum = np.zeros(1 << WORD_SIZE, dtype=np.float64)

    for _ in range(N_ITER):
        p0l = rng.integers(0, 1 << WORD_SIZE, size=NUM_PAIRS, dtype=np.uint16)
        p0r = rng.integers(0, 1 << WORD_SIZE, size=NUM_PAIRS, dtype=np.uint16)
        p1l = p0l ^ np.uint16(DIFF[0])
        p1r = p0r ^ np.uint16(DIFF[1])

        c0l, c0r = encrypt((p0l, p0r), round_keys)
        c1l, c1r = encrypt((p1l, p1r), round_keys)

        for s in range(0, 1 << WORD_SIZE, KEY_BATCH_SIZE):
            e = min(s + KEY_BATCH_SIZE, 1 << WORD_SIZE)
            keys = np.arange(s, e, dtype=np.uint16)
            score_accum[s:e] += score_keys(model, c0l, c0r, c1l, c1r, keys, SAMPLE_BATCH_SIZE, DEVICE)

    score_mean = score_accum / N_ITER
    ranking = np.argsort(-score_mean)
    rank_real = int(np.where(ranking == k_last)[0][0]) + 1

    np.savez(
        OUTPUT_PATH,
        score_mean=score_mean,
        ranking=ranking.astype(np.int32),
        k_last=np.uint16(k_last),
        rank_real=np.int32(rank_real),
        round_keys=round_keys,
    )

    print('k_last =', k_last)
    print('rank_real =', rank_real)
    print('saved to', OUTPUT_PATH)


if __name__ == '__main__':
    main()
