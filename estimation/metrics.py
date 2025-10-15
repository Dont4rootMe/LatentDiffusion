import torch
import spacy
from typing import Set
from evaluate import load
from nltk.util import ngrams
from collections import defaultdict
import spacy
import numpy as np


def compute_metric(metric_name, predictions, references, **kwargs):
    if metric_name == "mauve":
        return compute_mauve(predictions=predictions, references=references)
    elif metric_name == "div":
        return compute_diversity(all_texts_list=predictions)['diversity']
    elif metric_name == "mem":
        return compute_memorization(all_texts_list=predictions, human_references=references)
    elif metric_name == "ppl":
        return compute_ppl(predictions=predictions)
    else:
        raise Exception(f"Unknown metric: {metric_name}")
    

def filter_empty_texts(predictions, references):
    pred_list = []
    ref_list = []
    for i in range(len(predictions)):
        if predictions[i] and references[i]:
            pred_list.append(predictions[i])
            ref_list.append(references[i])
    return pred_list, ref_list


def compute_ppl(predictions, model_id='gpt2-large'):
    torch.cuda.empty_cache()

    predictions = [p for p in predictions if p]

    perplexity = load("perplexity", module_type="metric", model_id=model_id)
    ppl_list = perplexity.compute(
        predictions=predictions, 
        model_id=model_id, 
        device='cuda', 
        add_start_token=True,
    )["perplexities"]
    ppl_list = np.sort(ppl_list)
    quantile = 0.05
    a_min, a_max = int(quantile * len(ppl_list)), int((1 - quantile) * len(ppl_list))
    ppl_list = ppl_list[a_min: a_max]
    ppl = np.mean(ppl_list)
    return ppl


def compute_mauve(predictions, references, model_id='gpt2-large'):
    torch.cuda.empty_cache() 

    mauve = load("mauve")
    assert len(predictions) == len(references)

    if len(predictions) == 0:
        return 0

    predictions, references = filter_empty_texts(predictions, references)

    results = mauve.compute(
        predictions=predictions, references=references,
        featurize_model_name=model_id, device_id=0, verbose=False
    )

    return results.mauve


def compute_wordcount(all_texts_list):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=all_texts_list)
    return wordcount['unique_words']


def compute_diversity(all_texts_list):
    ngram_range = [2, 3, 4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in all_texts_list:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1 - len(ngram_sets[n])/ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1 - val)
    metrics['diversity'] = diversity
    return metrics


def compute_memorization(all_texts_list, train_unique_four_grams: Set[tuple[str]], n=4):
    tokenizer = spacy.load("en_core_web_sm").tokenizer

    total = 0
    duplicate = 0
    for sentence in all_texts_list:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in train_unique_four_grams:
                duplicate += 1

    return duplicate / total