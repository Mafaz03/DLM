
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

def text_to_token_ids(text, tokenizer, device="cpu"):
    # return torch.tensor(tokenizer.encode(text, allowed_special="<|endoftext|>")).unsqueeze(0)

    return torch.tensor(
                tokenizer.encode(
                        text,
                        allowed_special={"<|endoftext|>"}
                    ),
            device=device).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    
class GPT_Dataset_V1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        super().__init__()

        self.input_ids  = []
        self.target_ids = []

        self.token_ids = h = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(self.token_ids) - max_length, stride):
            self.input_ids.append(
                torch.tensor(self.token_ids[i:i + max_length])
            )

            self.target_ids.append(
                torch.tensor(self.token_ids[i + 1: i + max_length + 1])
            )
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(text, batch_size = 2, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 4):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPT_Dataset_V1(text, tokenizer, max_length, stride)
    # print(type(max_length), type(stride), type(text))
    
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        drop_last = drop_last, 
        num_workers = num_workers)
    
    return dataloader

# inferencing

def generate(model, tokens, max_new_tokens, context_size, temperature = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        tokens = tokens[:, -context_size:] # just in case it overflows
        logits = model(tokens)
        logits = logits[:, -1, :] # last context vector

        if top_k:
            top_k_logits, top_k_logits_idx = torch.topk(logits, top_k)
            logits = torch.where(
                condition = logits < top_k_logits[:, -1],
                input = torch.tensor(float("-inf")),
                other = logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
        else:
            idx_next = torch.argmax(torch.softmax(logits, dim = -1), dim = -1, keepdim=True)

        if idx_next == eos_id:
            break    
        tokens = torch.cat((tokens, idx_next), dim = 1)
    return tokens


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten(0,1)) # loss

def calc_loss_loader(dataloader, model, device, num_batches = None):
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len (dataloader))

    total_loss = 0
    for idx, (input_batch, target_batch) in enumerate(dataloader):
        if idx < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else: break
    return total_loss / num_batches

def generate_text_simple(model, tokens, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        tokens = tokens[:, -context_size:] # just in case it overflows
        logits = model(tokens)
        logits = logits[:, -1, :] # last context vector
        idx_next = torch.argmax(torch.softmax(logits, dim = -1), dim = -1, keepdim=True)
        tokens = torch.cat((tokens, idx_next), dim = 1)
    return tokens

def generate_and_print_samples(model, tokenizer, device, start_context, config, max_new_tokens = 50):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        generated_ids = generate_text_simple(model = model, tokens = encoded, max_new_tokens = max_new_tokens, context_size = config["context_length"])
    decoded = token_ids_to_text(generated_ids, tokenizer)
    print(decoded.replace("\n", " ")) # compacting
    model.train()

def train_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_itter, start_context, tokenizer, verbose = True, max_new_tokens = 50):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            optimizer.step()
            tokens_seen = input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device = device, num_batches = eval_itter)
                    val_loss   = calc_loss_loader(val_loader, model, device = device, num_batches = eval_itter)
                model.train()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )
                
        # print some samples
        if verbose:
            generate_and_print_samples(model = model,
                           tokenizer = tokenizer, device = device, start_context = start_context, max_new_tokens = max_new_tokens)
    return train_losses, val_losses, track_tokens_seen

