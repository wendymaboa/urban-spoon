"""Shared generation helpers for TRL experiment scripts."""

from __future__ import annotations

import torch


@torch.no_grad()
def get_value_estimates(model, query_tensors, response_tensors, device) -> torch.Tensor | None:
    """Return flattened value-head predictions for query+response sequences."""
    if not query_tensors:
        return None

    model.eval()
    values_list: list[torch.Tensor] = []
    for query, response in zip(query_tensors, response_tensors):
        input_ids = torch.cat([query.to(device), response.to(device)]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(output, 'value') and output.value is not None:
            values = output.value
        elif isinstance(output, tuple) and len(output) >= 3:
            values = output[2]
        else:
            continue
        values_list.append(values.reshape(-1).detach().cpu())

    model.train()
    if not values_list:
        return None
    return torch.cat(values_list)


def generate_batch(trainer, tokenizer, batch_prompts, device, max_new_tokens):
    """Sample one PPO batch; returns (queries, responses, decoded_responses)."""
    query_tensors = [
        tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=128,
        )['input_ids'].squeeze(0)
        for prompt in batch_prompts
    ]

    valid_queries, response_tensors = [], []
    for query in query_tensors:
        if query.numel() == 0:
            continue
        response = trainer.generate(
            query.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        valid_queries.append(query)
        response_tensors.append(response.squeeze(0)[len(query):].cpu())

    if not valid_queries:
        return [], [], []

    decoded = [tokenizer.decode(response, skip_special_tokens=True) for response in response_tensors]
    return valid_queries, response_tensors, decoded
