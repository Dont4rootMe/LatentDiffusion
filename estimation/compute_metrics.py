import argparse
import json
from transformers import AutoTokenizer
import numpy as np

from .metrics import compute_metric

def truncate_text(texts, max_length, min_required_length):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_texts = tokenizer(texts, padding=False, truncation=True, max_length=max_length)["input_ids"]
    tokenized_texts = [text for text in tokenized_texts if len(text) >= min_required_length]
    truncated_texts = tokenizer.batch_decode(tokenized_texts, skip_special_tokens=True)
    return truncated_texts


def main(args):
    num_times = args.num_times
    max_length = args.max_length
    num_samples = args.num_samples
    min_required_length = args.min_required_length

    with open(args.pred_file, "r") as f:
        pred_data = json.load(f)

    predictions = [pred["GEN"] for pred in pred_data]
    predictions = truncate_text(predictions, max_length, min_required_length)

    references = [pred["TRG"] for pred in pred_data]
    references = truncate_text(references, max_length, min_required_length)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of references: {len(references)}")

    import random
    
    # Randomly sample num_samples texts if there are more than num_samples

    metrics_dict = {
        "ppl": [],
        "mauve": [],
        "div": [],
    }

    for i in range(num_times):
        indices = random.sample(range(len(predictions)), num_samples)
        batch_predictions = [predictions[i] for i in indices]
        indices = random.sample(range(len(references)), num_samples)
        batch_references = [references[i] for i in indices]

        for metric_name in ["ppl", "mauve", "div"]:
            metrics_dict[metric_name].append(compute_metric(
                metric_name, 
                predictions=batch_predictions, 
                references=batch_references,
            ))

    print(f"Pred_file: {args.pred_file}")
    for key, value in metrics_dict.items():
        print(f"{key}: {np.mean(value):0.5f} ± {np.std(value):0.5f}")
        print(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--num_times", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--min_required_length", type=int, default=500)
    args = parser.parse_args()

    main(args)

"""
CUDA_VISIBLE_DEVICES=0 \
python -m estimation.compute_metrics \
    --pred_file ./generated_texts/diffusion-rocstories-16-d=5-v2.3.4/200000-N=200-len=9216.json \
    --num_times 3 \
    --num_samples 1000 \
    --min_required_length 0 \
    --max_length 128

"""