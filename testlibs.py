# Save as test_gpt2.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading GPT-2 small...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e6:.0f} MB")

# Quick generation test
inputs = tokenizer("Tell me something helpful:", return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
print("Generated:", tokenizer.decode(output[0], skip_special_tokens=True))
print("GPT-2 test passed.")