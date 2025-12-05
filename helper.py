
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

def text_to_token_ids(text, tokenizer):
    # return torch.tensor(tokenizer.encode(text, allowed_special="<|endoftext|>")).unsqueeze(0)

    return torch.tensor(
                tokenizer.encode(
                        text,
                        allowed_special={"<|endoftext|>"}
                    )
            ).unsqueeze(0)

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