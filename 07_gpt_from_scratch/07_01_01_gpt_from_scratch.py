# %%
import numpy as np
import pandas as pd
# %%
with open("../datasets/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# %%
print(f"length of dataset in characters: {len(text)}")
# %%
## first 1000 characters
print(text[:1000])
# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)
# %%
# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] ##Encoder: Take a string and output a list of characters
decode = lambda l: "".join([itos[i] for i in l]) ## Decoder: take a list of integers and output a string

string = "hii there"
print(encode(string))
print(decode(encode(string)))
# %%
## pip3 install torch torchvision torchaudio
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
# %%
## Splitting up the data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
# %%
## We train a neural network by taking the data
## one chunk at a time.
block_size = 8
train_data[:block_size+1]
# %%
## mini batch
torch.manual_seed(1337)
batch_size = 4 # how many independent sequence will we process in parallel
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x,y
# %%
xb,yb = get_batch("train")
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('----------')
# %%
## What is input and what is the target?
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is {context.tolist()} the target is {target.tolist()}")
# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logic for the next token
        # from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
def generate(self, idx, max_new_tokens):
    # idx is a (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
        # get the prediction
        logits, loss = self(idx)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B,C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=1) # (B,C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
    return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(out.shape)