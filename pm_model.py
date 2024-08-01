import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from collections import deque

class LoRA(nn.Module):
    def __init__(self, hidden_size, r):
        super(LoRA, self).__init__()
        self.B = nn.Linear(hidden_size, r, bias=False)
        self.A = nn.Linear(r, hidden_size, bias=False)
        
    def forward(self, x):
        return x + self.B(x) @ self.A.weight.T

class Router(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super(Router, self).__init__()
        self.linear = nn.Linear(hidden_size, num_experts)
        
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

class PMoE(nn.Module):
    def __init__(self, base_model_path, num_experts=8, tau=24, r=4):
        super(PMoE, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.num_experts = num_experts
        self.tau = tau

        # Freeze shallow layers
        for param in self.base_model.model.layers[:tau].parameters():
            param.requires_grad = False

        # Initialize LoRA experts for deep layers
        hidden_size = self.base_model.config.hidden_size
        self.deep_layers = nn.ModuleList([LoRA(hidden_size, r) for _ in range(len(self.base_model.model.layers) - tau)])
        self.experts = nn.ModuleList([
            nn.ModuleList([
                LoRA(hidden_size, r)
                for _ in range(len(self.base_model.model.layers) - tau)
            ])
            for _ in range(num_experts)
        ])

        # Router
        self.router = Router(hidden_size, num_experts)

    def forward(self, input_ids, attention_mask=None):
        # Shallow layers (up to tau)
        shallow_output = self.base_model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_state = shallow_output.hidden_states[self.tau]

        # Router
        router_logits = self.router(hidden_state)
        router_probs = F.softmax(router_logits, dim=-1)

        # Deep layers (after tau)
        for i in range(len(self.deep_layers)):
            expert_outputs = []
            for expert in self.experts:
                expert_output = expert[i](hidden_state)
                expert_outputs.append(expert_output)

            # Combine expert outputs
            combined_output = sum([output * prob.unsqueeze(-1) for output, prob in zip(expert_outputs, router_probs.unbind(-1))])

            # Apply the deep layer
            hidden_state = self.deep_layers[i](combined_output)

        # Final output
        lm_logits = self.base_model.lm_head(hidden_state)

        return lm_logits, router_logits

    def generate(self, input_ids, max_length=50, **kwargs):
        attention_mask = torch.ones_like(input_ids)
        generated = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            outputs, _ = self.forward(generated, attention_mask)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(-1))], dim=-1)

        return generated

# Custom dataset to handle replay mechanism
class CustomDataset(Dataset):
    def __init__(self, current_data, replay_buffer):
        self.data = current_data
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.data) + len(self.replay_buffer)

    def __getitem__(self, idx):
        if idx < len(self.data):
            return self.data[idx]
        else:
            return self.replay_buffer[idx - len(self.data)]

# Load the TRACE dataset and create synthetic tasks
def load_trace_dataset(task_id, num_tasks=8):
    dataset = load_dataset("trace", split="train")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Split dataset into num_tasks parts
    task_size = len(encoded_dataset) // num_tasks
    task_dataset = encoded_dataset.select(range(task_id * task_size, (task_id + 1) * task_size))
    return task_dataset

# Training function
def train_pm_model(pm_model, num_tasks=8, learning_rate=3e-4, batch_size=8, epochs=1, replay_ratio=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pm_model.to(device)
    
    optimizer = AdamW(pm_model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epochs * num_tasks)
    
    replay_buffer = deque(maxlen=int(replay_ratio * len(load_trace_dataset(0, num_tasks))))  # Adjust size based on replay ratio

    for task_id in range(num_tasks):
        dataset = load_trace_dataset(task_id, num_tasks)
        train_dataloader = DataLoader(CustomDataset(dataset, replay_buffer), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            pm_model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs, router_logits = pm_model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))

                # Auxiliary loss for router
                router_targets = torch.full((router_logits.size(0),), task_id, dtype=torch.long).to(device)
                router_loss = F.cross_entropy(router_logits, router_targets)
                total_loss = loss + router_loss

                total_loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print(f"Task: {task_id + 1}/{num_tasks}, Epoch: {epoch + 1}, Loss: {total_loss.item()}")

        # Add current task data to replay buffer
        for item in dataset:
            replay_buffer.append(item)

    # Evaluation metrics
    overall_performance = 0
    backward_transfer = 0
    for t in range(num_tasks):
        dataset = load_trace_dataset(t, num_tasks)
        test_dataloader = DataLoader(dataset, batch_size=batch_size)
        pm_model.eval()
        task_performance = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs, _ = pm_model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
                task_performance += loss.item()
        
        task_performance /= len(test_dataloader)
        overall_performance += task_performance
        if t < num_tasks - 1:
            backward_transfer += task_performance - overall_performance / (t + 1)
        
    overall_performance /= num_tasks
    backward_transfer /= num_tasks - 1
    
    print(f"Overall Performance (OP): {overall_performance}")
    print(f"Backward Transfer (BWT): {backward_transfer}")

# Main execution
if __name__ == "__main__":
    base_model_path = "meta-llama/Meta-Llama-3.1-8B"
    tasks = [f"task{i}" for i in range(1, 9)]  # Replace with actual task names

    pm_model = PMoE(base_model_path)
    train_pm_model(pm_model)
