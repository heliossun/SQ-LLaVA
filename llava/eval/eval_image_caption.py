from evaluate import load
from datasets import Dataset, load_dataset
from nlgmetricverse import NLGMetricverse
from nlgmetricverse import load_metric
import argparse
import torch
import os
import json
from tqdm import tqdm
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def evaluate_caption(metrics:list,predictions:list=[],references:list=[]):
    # results=[]
    # if data == "coco-kapthy":
    #     dt = load_dataset("yerevann/coco-karpathy",split='validation')
    #     reference = dt['sentences']

    # bleu = load("bleu")
    # bleu_results = bleu.compute(predictions=predictions, references=reference)   #prediction: list of strings, reference: list of list
    scorer = NLGMetricverse(metrics=metrics)
    scores = scorer(predictions=predictions, references=references,reduce_fn="max")
    #scores = scorer.compute(predictions=predictions, references=references,reduce_fn="max")
    return scores

def eval_model(args):
    # Model

    answers = [json.loads(q) for q in open(os.path.expanduser(args.answers_file), "r")]
    answers = get_chunk(answers, args.num_chunks, args.chunk_idx)
    
    anno = json.load(open(os.path.join(args.annotation),'r'))
    references=[]        
    predictions=[]
    for i,line in enumerate(tqdm(answers)):
        references.append(anno[i]['caption'])
        predictions.append(line["text"])
    metrics = [
        load_metric("bleu"),
        load_metric("meteor"),
        load_metric("cider")]
    #print(references[0])
    #print(predictions[0])
    scores=evaluate_caption(metrics=metrics, predictions=predictions,references=references)
    print(f"bleu@4:{scores['bleu']['score']}, meteor: {scores['meteor']['score']}, CiDer: {scores['cider']['score']}")
    #print(scores)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
