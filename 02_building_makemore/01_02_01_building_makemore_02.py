# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %%
words = open("../datasets/names.txt", "r").read().splitlines()
words[:8]
# %%
len(words)
# %%
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
# %%
# building the dataset
block_size = 3; # context length: how many characters do we take to predict the next one
X, Y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size
    for ch in w+'.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '-->', itos[ix])
        context = context[1:] + [ix]
X = torch.tensor(X)
Y = torch.tensor(Y)
# %%
embedding_size = 2
C = torch.randn((len(chars),embedding_size))
# %%
# this is the embedding of 5th chracter
C[5]
# %%
# we are retrieving the embedding of fifth character, but by doing a matrix multiplication
F.one_hot(torch.tensor(5), num_classes=27).float() @ C
# %%
emb = C[X]
emb.shape
# %%
num_neurons_layer_1 = 100
W1 = torch.randn((block_size*embedding_size,num_neurons_layer_1))
b1 = torch.randn(num_neurons_layer_1)
# %%
# One way to do the multiplication with W1 is to match the dimensions,
# using unbind and cat
torch.cat(torch.unbind(emb,1),1) @ W1 +b1
# %%
# another way to do the flatteing and the multiplications is using the view
emb.view(32,6) @ W1 + b1
# %%
h = torch.tanh(emb.view(-1,6)@ W1 + b1)
# %%
# now we are creating the output layer. Our output layer is softmax
# to select one of the most lilkely characters
W2 = torch.randn((W1.shape[1],len(chars)))
b2 = torch.randn(len(chars))
# %%
logits = h @ W2 + b2
# %%
counts = logits.exp()
# %%
prob = counts / counts.sum(1, keepdims=True)
# %%
Y
# tensor([ 5, 13, 13,  1,  0, 15, 12,  9, 22,  9,  1,  0,  1, 22,  1,  0,  9, 19,
#          1,  2,  5, 12, 12,  1,  0, 19, 15, 16,  8,  9,  1,  0])
# %%
prob[torch.arange(32),Y]
# think that you want to get the probability of 5th character from 
# the 0 th row, then the 13th character from the 1st row, then
# the 13th character from the 2nd row.
# We can do this in two ways, one is getting each element one at a time
# >>> prob[(0,5)]
# tensor(4.2094e-15)
# or creating a two lists, first list is for first dimension, and 
# the second is for the second dimension
# >>> prob[[0],[5]]
# tensor([4.2094e-15])
# Similary:
# >>> prob[[0, 1, 2],[5, 13, 13]]
# tensor([4.2094e-15, 1.5536e-09, 8.4170e-12])
# %%
# Now creating the loss
loss = -prob[torch.arange(32),Y].log().mean()
loss
# %%
## Now making the whole code respectable
# building the dataset
block_size = 3; # context length: how many characters do we take to predict the next one
X, Y = [], []
for w in words:
    #print(w)
    context = [0] * block_size
    for ch in w+'.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join(itos[i] for i in context), '-->', itos[ix])
        context = context[1:] + [ix]
X = torch.tensor(X)
Y = torch.tensor(Y)
X.shape, Y.shape # dataset
# %%
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator = g)
W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100,27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
  p.requires_grad = True
# %%
sum(p.nelement() for p in parameters) # number of parameters iin total
# %%
emb = C[X] # (32,3,2)
h = torch.tanh(emb.view(-1,6) @ W1 + b1) # (32,100)
logits = h @ W2 + b2 # (32,27)
counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
loss = -prob[torch.arange(32), Y].log().mean()
loss

# %%
# We can also do the loss calculation using builtin cross_entropy funtion
F.cross_entropy(logits,Y)
# It is preferred because internally pytorch will make sure even
# with very large number it will not go to inf or produce NAN
# https://youtu.be/TCH_1BHY58I?si=4MHmit48r-et52mU 
# time 37:50
# %%
# batch_size = 32
# for _ in range(1000):
#     # minibatch construction
#     ix = torch.randint(0, X.shape[0], (batch_size,))

#     # forward pass
#     emb = C[X[ix]] # (32,3,2)
#     h = torch.tanh(emb.view(-1,6) @ W1 + b1) # (32,100)
#     logits = h @ W2 + b2 # (32,27)
#     loss = F.cross_entropy(logits,Y[ix]).float()
#     print(loss.item())

#     # backward pass
#     for p in parameters:
#         p.grad = None
#     #loss.requires_grad = True
#     loss.backward()
    
#     # update
#     for p in parameters:
#         p.data += -0.1 * p.grad
# %%
emb = C[X] # (32,3,2)
h = torch.tanh(emb.view(-1,6) @ W1 + b1) # (32,100)
logits = h @ W2 + b2 # (32,27)
loss = F.cross_entropy(logits,Y).float()
print(loss.item())
# %%
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
# %%
lri = []
lossi = []
batch_size = 32
for i in range(1000):
    # minibatch construction
    ix = torch.randint(0, X.shape[0], (batch_size,))

    # forward pass
    emb = C[X[ix]] # (32,3,2)
    h = torch.tanh(emb.view(-1,6) @ W1 + b1) # (32,100)
    logits = h @ W2 + b2 # (32,27)
    loss = F.cross_entropy(logits,Y[ix]).float()
    print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    #loss.requires_grad = True
    loss.backward()
    
    # update
    lr = lrs[i]
    for p in parameters:
        p.data += -0.1 * p.grad

    # track stats
        lri.append(lrs[i])
        lossi.append(loss.item())
# %%
plt.plot(lri,lossi)
# %%
