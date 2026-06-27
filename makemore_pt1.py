# %%
# 0

with open("names.txt", "r") as file:
    words = file.read().splitlines()

# %%
# 1

b = {}
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(w, w[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
sorted(b.items(), key=lambda kv: -kv[1])
# %%
# 2

import torch

a = torch.zeros((3, 5), dtype=torch.int32)
# %%
# 3

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {s: i for i, s in stoi.items()}

# %%
# 4

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# %%
# 5

p = N[0].float()
p /= p.sum()

# %%
# 6

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
p = torch.rand(3, generator=g)
p /= p.sum()
p

# %%
# 7

g = torch.Generator().manual_seed(2147483647)
N += 1
n_normalised = N / N.sum(dim=1, keepdim=True)
ix = 0
word = ""
ch = ""
while ch != ".":
    p = n_normalised[ix]
    ix = int(torch.multinomial(p, num_samples=1, replacement=True, generator=g).item())
    ch = itos[ix]
    word += ch

# %%
# 8
log_likelihood = 0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = n_normalised[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood -= logprob
        n += 1

log_likelihood /= n
log_likelihood

# %%
# 9

xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(num)
xs.shape
ys.shape
# %%
# 9

import torch.nn.functional as F

W = torch.randn((27, 27), requires_grad=True)
loss = 0

for epoch in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    print(loss.item())
    W.grad = None
    loss.backward()
    assert W.grad != None
    W.data += -50 * W.grad

# %%
# 9
