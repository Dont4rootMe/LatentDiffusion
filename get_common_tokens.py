import os
import sys
import torch
import argparse
import wandb
from datetime import timedelta
from transformers import AutoTokenizer
import numpy as np
import json
import pandas as pd

from encoder_trainer import EncoderTrainer
from utils import set_seed
from encoder_config import create_config


if __name__ == '__main__':
    config = create_config(None)
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    max_length = config.data.max_sequence_len

    result_path = "text_reconstruction/result-[4].json"
    index = 6

    with open(result_path, "r") as f:
        data = json.load(f)

    target_tokens = [int(t) for t in data[index]["target_tokens"].split(",")]
    prediction_tokens = [int(t) for t in data[index]["prediction_tokens"].split(",")]

    target_symbols = [tokenizer.decode(t, skip_special_tokens=False) for t in target_tokens]
    prediction_symbols = [tokenizer.decode(t, skip_special_tokens=False) for t in prediction_tokens]

    # print(len(target_symbols))
    # print(len(prediction_symbols))

    # print(target_symbols)
    # print(prediction_symbols)

    table = pd.DataFrame.from_dict(
        {
            "target_symbols": target_symbols,
            "prediction_symbols": prediction_symbols,
            "is_same": [(t1 == t2) for t1, t2 in zip(target_tokens, prediction_tokens)],
        }
    )
    table.to_csv("common_tokens.csv", sep=',', index=False, encoding='utf-8')

    accuracy = (np.array(target_tokens) == np.array(prediction_tokens)) * 1.
    print(np.mean(accuracy))