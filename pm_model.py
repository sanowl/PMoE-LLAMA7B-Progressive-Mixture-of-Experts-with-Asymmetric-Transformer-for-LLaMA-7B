import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader

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
        self.base_model = LlamaForCausalLM.from_pretrained(base_model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        self.num_experts = num_experts
        self.tau = tau

        # Freeze shallow layers
        for param in self.base_model.model.layers[:tau].parameters():
            param.requires_grad = False

        # Initialize LoRA experts for deep layers
        hidden_size = self.base_model.config.hidden_size
        self.shallow_layers = nn.ModuleList([LoRA(hidden_size, r) for _ in range(tau)])
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
        shallow_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get the hidden state at the tau layer
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

        return lm_logits

    def generate(self, input_ids, max_length=50, **kwargs):
        attention_mask = torch.ones_like(input_ids)
        generated = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(generated, attention_mask)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(-1))], dim=-1)

        return generated

# Load the TRACE dataset
def load_trace_dataset():
    dataset = load_dataset("trace", split="train")
    tokenizer = LlamaTokenizer.from_pretrained("huggingface/llama-7b")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return encoded_dataset

# Training function
def train_pm_model(pm_model, dataset, learning_rate=3e-4, batch_size=8, epochs=1):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(pm_model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pm_model.to(device)

    for epoch in range(epochs):
        pm_model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = pm_model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Main execution
if __name__ == "__main__":
    base_model_path = "huggingface/llama-7b"
    pm_model = PMoE(base_model_path)
    trace_dataset = load_trace_dataset()
    train_pm_model(pm_model, trace_dataset)
