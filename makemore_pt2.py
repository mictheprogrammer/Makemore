# %%
# 0

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# %%
# 1

with open("names.txt", "r") as file:
    words = file.read().splitlines()

len(words)

# %%
# 2

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
print(itos)


# %%
# 3
def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    Y = torch.tensor(Y)
    X = torch.tensor(X)
    return X, Y


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Y_te = build_dataset(words[n2:])

# %%
# 4

C = torch.randn((27, 10))
# %%
# 5

W1 = torch.randn((30, 200)) * (5 / 3) / (30**0.5)
W2 = torch.randn(200, 27) * 0.01
b2 = torch.zeros(27)
bngain = torch.ones((1, 200))
bnbias = torch.zeros((1, 200))
bnmean_running = torch.zeros((1, 200))
bnstd_rnning = torch.ones((1, 200))
parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

# %%
# 6
loglr = torch.linspace(-3, 0, 1000)
lrlst = 10**loglr

# %%
# 7

lri = []
lossi = []
stepi = []
for i in range(300):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C[Xtr[ix]]
    h = emb.view(emb.shape[0], 30) @ W1
    bnmeani = h.mean(0, keepdim=True)
    bnstdi = h.std(0, keepdim=True)
    h = bngain * (h - bnmeani) / bnstdi + bnbias
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_rnning = 0.999 * bnstd_rnning + 0.001 * bnstdi
    h = torch.tanh(h)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad  # type: ignore
    stepi.append(i)
    lossi.append(loss.item())

# %%
# 8

with torch.inference_mode():
    emb = C[Xdev]
    h = emb.view(-1, 30) @ W1
    h = bngain * (h - bnmean_running) / bnstd_rnning + bnbias
    h = torch.tanh(h)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(loss)


# %%
# 9

with torch.inference_mode():
    block_size = 3
    for _ in range(20):
        out = []
        context = [0] * block_size
        ix = 1
        while ix != 0:
            emb = C[torch.tensor(context)]
            h = emb.view(1, -1) @ W1
            h = bngain * (h - bnmean_running) / bnstd_rnning + bnbias
            h = torch.tanh(h)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
        print("".join(itos[i] for i in out))


# %%
# 10


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = torch.randn((in_features, out_features)) / in_features**0.5
        self.bias = torch.zeros(out_features) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_std = torch.ones(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


block_size = 3
n_embd = 10
n_hidden = 100
vocab_size = len(itos)

C = torch.randn((vocab_size, n_embd))
layers = [
    Linear(n_embd * block_size, n_hidden),
    Tanh(),
    BatchNorm1d(n_hidden),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
]

with torch.no_grad():
    layers[-1].weight *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5 / 3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

# %%
# 11

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < max_steps / 2 else 0.01
    for p in parameters:
        assert p.grad is not None
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f"{loss.item():.4f}")
    lossi.append(loss.log10().item())
    break

# %%
# 11
