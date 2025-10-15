import torch
from tqdm import tqdm
from evaluate import load
import json
import argparse

@torch.no_grad()
def get_ppl(data):
    model_id = 'gpt2-large'
    perplexity = load("perplexity", module_type="metric", model_id=model_id)

    ppls = []

    texts = []

    end_indices = [0]

    for i in tqdm(range(len(data["alpha"]))):
        pred_text = data["pred_text"][i]
        pred_text = [t for t in pred_text if t != ""]
        texts.extend(pred_text)
        end_indices.append(len(texts))

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        ppl = perplexity.compute(
            predictions=texts, 
            model_id=model_id, 
            device='cuda', 
            add_start_token=True,
        )["perplexities"]

    ppls = [ppl[s:e] for s, e in zip(end_indices[:-1], end_indices[1:])]
    
    return ppls


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--version", type=str, default="v1.0")
    args = args.parse_args()

    with open(f"./experiments/interpolation-all_times-diffusion-rocstories-16-d=5-{args.version}.json", "r") as f:
        data_v10_interpolation = json.load(f)

    ppl_v10 = {}

    for t in data_v10_interpolation.keys():
        ppl_v10[t] = get_ppl(data_v10_interpolation[t])
    
    json.dump(ppl_v10, open(f"./experiments/interpolation-all_times-diffusion-rocstories-16-d=5-{args.version}-ppl.json", "w"))

"""
CUDA_VISIBLE_DEVICES=0 \
python -m experiments.interpolation_ppl \
--version v1.0
"""