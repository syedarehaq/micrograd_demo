# %%
import numpy as np
# %%
words = open("../datasets/names.txt", "r").read().splitlines()
# %%
len(words)
# %%
max([len(w) for w in words])
# %%
#min([len(w)] for w in words)
min([len(w) for w in words])
# %%
b = {}
for w in words:
    chs = ["<s>"] + list(w) + ["<E>"]
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram, 0) + 1
        #print(ch1, ch2)
# %%
sorted(b.items(), key=lambda x: -x[1])
# %%
import torch
# %%
chars = sorted(set([c for w in words for c in w]))
stoi = {c:i+1 for i,c in enumerate(chars)}
stoi["."] = 0
itos = {i:c for c,i in stoi.items()}
# %%
N = torch.zeros((28,28), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1
# %%
import matplotlib.pyplot as plt
# %%
plt.figure(figsize=(16,16))
plt.imshow(N, cmap="Blues")
for i in range(len(stoi)):
    for j in range(len(stoi)):
        chstr = itos[i] + itos[j]
        plt.text(j,i, chstr, ha="center", va="bottom", color="grey")
        plt.text(j,i, N[i,j].item(), ha="center", va="top", color="gray")
plt.axis("off")
# %%
p = N[0].float()
p = p/p.sum()
p
# %%
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[ix]
# %%
p = torch.rand(3, generator=g)
p = p/p.sum()
# %%
P = (N+1).float()
P /= P.sum(dim = 1, keepdim=True)
# Read through the broadcasting semantics in pytorch
# %%
P.sum(dim=1,keepdim=True)
# %%
for i in range(100):
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        p = p/p.sum()
        #p = torch.ones(27) / 27.0
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        #print(itos[ix])
        if ix == 0:
            break
    print("".join(out))
# %%
log_likelihood = 0.0
n = 0
for w in words[:3]:
    chs = ["."] + list(w) + ["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        n += 1
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        print(f"{ch1}{ch2}: {prob:.4f} {logprob: .4f}")
print(f"{log_likelihood.item()=}")
nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/n}")
# %%
## Loss function: Low is good, because we minimize the loss

# GOAL: Maximize the likelihood of the data w.r.t model parameters (statistical modeling)
# equivalent to maxmizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# %%
## Second part: Neural network approach
# Create a training set of bigrams (x,y)
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
# %%
import torch.nn.functional as F
# %%
xenc = F.one_hot(xs, num_classes = 27).float()
# %%
xenc.dtype
# %%
## When we are plugging numbers to a neural net we do not want them to be integers
## we want them to be floating point number
W = torch.randn(27,27, generator=g, requires_grad = True)
# %%
## In matrix multiplication the middle dimensions squeezes because they multiply between
## themselves and then we are left with their addition

## A single neuron linear layer would be at least a X dimensional vector when the 
## input is X dimensional. If there are multiple neurons e.g. N neurons, then it would
## be a X by N dimensional vector.

## Softmax converts a vector of numbers into a vector of probability

## Logit, you cna think about the logit as the log count
## If you exponentiate it, you can think of them as something equivalent to the 
## actual count as they are positive numbers
# %%
logits = xenc @ W # log counts
counts = logits.exp() # equivalent of N matrix
# probability is the counts normalizedd
prob = counts/counts.sum(1, keepdims=True)
# %%
# Forward pass
for epoch in range(1000):
    xenc = F.one_hot(xs, num_classes=27).float() # input to the network, one hot encoding
    logits = xenc @ W # Predict log counts
    counts = logits.exp() # counts, equivalent to N
    probs = counts / counts.sum(1, keepdim=True) # probabilities for next chracter
    # Can we optimize such that probability of seeing 5 is high for the 0th input?
    # probability of seeing 13 is high if for the 1th input, which is
    #probs[0,5], probs[1,13], probs[2,13], probs[3,1], probs[4,0]
    #torch.arange(5)
    loss=-probs[torch.arange(num), ys].log().mean()
    #loss
    print(loss.item())

    # backward pass
    W.grad = None # set to zero gradient
    loss.backward()
    W.data += -2 * W.grad

# %%
g = torch.Generator().manual_seed(2147483647)
for i in range(50):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes = 27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts/counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples = 1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print("".join(out))

# %%
