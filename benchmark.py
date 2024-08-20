import torch
import time
import psutil
import math
import gzip
import numpy as np
from bitnet import BitNetTransformer
from bitnet.at import AutoregressiveWrapper
from torch.utils.data import DataLoader, Dataset

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

def load_model(model_path):
    model = BitNetTransformer(num_tokens=256, dim=512, depth=8)
    model = AutoregressiveWrapper(model, max_seq_len=1024)
    model.load_state_dict(torch.load(model_path))
    return model

def benchmark_latency(model, dataloader):
    print(f"Benchmarking latency...")
    model.eval()
    
    latencies = []
    with torch.no_grad():
        print(f"Starting benchmark...")
        print(f"Dataloader: {dataloader}")
        batch_count = 0
        for batch in dataloader:
            print(f"Batch {batch_count}: {batch}")
            start_time = time.time()
            _ = model(batch)
            end_time = time.time()
            latencies.append(end_time - start_time)
            batch_count += 1
    return sum(latencies) / len(latencies)

def benchmark_memory_usage(model, dataloader):
    print(f"Benchmarking memory usage...")
    model.eval()
    max_memory = 0
    with torch.no_grad():
        print(f"Starting benchmark...")
        print(f"Dataloader: {dataloader}")
        batch_count = 0
        for batch in dataloader:
            print(f"Batch {batch_count}: {batch}")
            _ = model(batch)
            memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
            max_memory = max(max_memory, memory)
            batch_count += 1
    return max_memory

def benchmark_ppl(model, dataloader):
    print(f"Benchmarking perplexity...")
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        print(f"Starting benchmark...")
        print(f"Dataloader: {dataloader}")
        batch_count = 0
        for batch in dataloader:
            print(f"Batch {batch_count}: {batch}")
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            outputs = model(input_ids)
            
            # Check the shape of outputs
            print(f"Outputs shape: {outputs.shape}")
            
            # Ensure outputs and target_ids have the correct shape
            if outputs.dim() == 3:
                loss = F.cross_entropy(outputs.transpose(1, 2), target_ids, reduction='sum')
            else:
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target_ids.view(-1), reduction='sum')
            
            total_loss += loss.item()
            total_tokens += target_ids.numel()
            batch_count += 1
    return math.exp(total_loss / total_tokens)

if __name__ == "__main__":
    model_paths = ["bitnet_final_model.pth"]  # Add your model paths here
    
    # Prepare enwik8 data (similar to train.py)
    with gzip.open("./data/enwik8.gz") as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        _, vaX = np.split(X, [int(90e6)])  # We'll use the validation set for benchmarking
        data_val = torch.from_numpy(vaX)

    # Create dataset and dataloader
    SEQ_LEN = 1024
    dataset = TextSamplerDataset(data_val, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Rest of the benchmarking code remains the same
    for model_path in model_paths:
        print(f"Benchmarking model: {model_path}")
        print(f"Loading model...")
        model = load_model(model_path)
        print(f"Model loaded successfully.")
        
        print(f"Benchmarking latency...")
        latency = benchmark_latency(model, dataloader)
        print(f"Benchmarking memory usage...")
        memory_usage = benchmark_memory_usage(model, dataloader)
        print(f"Benchmarking perplexity...")
        ppl = benchmark_ppl(model, dataloader)
        
        print(f"Model: {model_path}")
        print(f"Average Latency: {latency*1000:.2f} ms")
        print(f"Max Memory Usage: {memory_usage:.2f} MB")
        print(f"Perplexity: {ppl:.2f}")
        print("--------------------")