import torch
from torch.nn import functional as F
from torch import nn
import numpy as np

# ------
block_size = 256
batch_size = 64
head_size = 16
n_embed = 384
dropout_rate = 0.2
n_head = 6
n_layer = 6
lr = 3e-4
max_itter = 5_000
# ------

# Dataset
with open("shakesphere.txt", 'r', encoding='utf-8') as f:
    ds_txt = f.read()
    

chars = sorted(list(set(ds_txt)))
vocab_size = len(chars)

encoder_map = { ch:i for i,ch in enumerate(chars) }
decoder_map = { i:ch for i,ch in enumerate (chars)}

encoder = lambda text: [encoder_map[tex] for tex in text]
decoder = lambda lis: ''.join([decoder_map[li] for li in lis])


data = torch.tensor(encoder(ds_txt), dtype=torch.long)
n = int(.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    random_locations = torch.randint(
                            low=0,
                            high=len(data) - block_size, # so it doesnt go out of bounds
                            size=(batch_size,)
                        )
    
    inputs  = torch.stack([data[i     : i+block_size] for i in random_locations])
    targets = torch.stack([data[i + 1 : i+block_size+1] for i in random_locations])
    return inputs, targets

# loss
@torch.no_grad()
def average_loss(model, eval_itter):
    model.eval()
    avg_loss = {'train': None, 'val': None}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_itter)
        for i in range(eval_itter):
            inputs, targets = get_batch(split)
            logits, loss = model(inputs, targets)
            losses[i] = loss.item()
        avg_loss[split] = losses.mean()
    model.train()
    return avg_loss

# Train loop and inference
def train(model, epochs = 100):
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    for step in range(epochs):
        inputs, targets = get_batch('train')
        losses_dict = average_loss(model, eval_itter = 10)
        logits, loss = model(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step: {step+1} | training loss: {losses_dict['train']:.4f} | validation loss: {losses_dict['val']:.4f}")

    return model

def generate_bs(model, max_tokens = 1000):
    inputs, targets = get_batch('val')
    generated_outputs = model.generate(inputs, max_tokens)
    for gen_out in generated_outputs:
        print(decoder(gen_out.tolist()))
        print("----")


# Self attention

class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key   = torch.nn.Linear(n_embed, head_size, bias = False) # (n_embedC, head_size)
        self.query = torch.nn.Linear(n_embed, head_size, bias = False) # (n_embedC, head_size)
        self.value = torch.nn.Linear(n_embed, head_size, bias = False) # (n_embedC, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                    # (B, T, C) = (B, T, head_size)
        q = self.query(x)                  # (B, T, C) = (B, T, head_size)
        v = self.value(x)                  # (B, T, C) = (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) . (B, C, T) = (B, T, T)
        wei_masked = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei_masked, 1)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
    
class Feedforward(torch.nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.block = torch.nn.Sequential(
                 torch.nn.Linear(n_embed, 4 * n_embed),
                 torch.nn.ReLU(),
                 torch.nn.Linear(4 * n_embed, n_embed),
                 torch.nn.Dropout(dropout_rate)
                 )
     
    def forward(self, x):
        return self.block(x)

class Multihead_attention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.mha_heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = torch.nn.Linear(n_embed, n_embed)
        self.dropout =  torch.nn.Dropout(dropout_rate)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.mha_heads], dim = -1)
        # print(out.shape)
        out = self.dropout(self.projection(out))
        return out
        

class Block(torch.nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()

        head_size = n_embed // n_head
        self.self_multi_head_attention = Multihead_attention(n_head, head_size) # one head of self-attention. (B,T, C)
        self.ff = Feedforward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.self_multi_head_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BigramLanguage(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = torch.nn.Embedding(block_size, n_embed)

        # self.self_attention_head = Head(n_embed)
        self.self_multi_head_attention = Multihead_attention(4, n_embed//4)
        
        self.lm_head = torch.nn.Linear(n_embed, vocab_size)

        self.Blocks = torch.nn.Sequential(
            * [Block(n_embed, n_head=n_head) for _ in range(n_layer)]
            )
        
        self.ln_f = torch.nn.LayerNorm(n_embed)

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            cond_idx = idx[:, -block_size:]
            logits, _ = self(cond_idx)
            # return logits
            logits = logits[:, -1, :] # (B, T, C) -> (B, C)
            prob = F.softmax(logits, dim=-1) # probabilities
            idx_next = torch.multinomial(prob, num_samples=1) # Sampling (B, C) -> (B, 1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
    
    def forward(self, idx, targets = None):
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx)                         # (B, T, n_embed)
        positional_embedding = self.positional_embedding(torch.arange(T))     # (T, n_embed)
        x = token_embed + positional_embedding
        x = self.Blocks(x)
        x = self.ln_f(x)
        # x = self.self_attention_head(x)                                     # one head of self-attention. (B,T, C)
        # x = self.self_multi_head_attention(x)                               # one head of self-attention. (B,T, C)
        # x = self.ff(x)
        logits = self.lm_head(x)                                              # (B, T, vocab_size)
        

        if targets != None:  
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
            
        else: loss = None

        return logits, loss
    
model = BigramLanguage(vocab_size)
model = train(model, max_itter)