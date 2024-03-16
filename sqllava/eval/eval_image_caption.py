
import argparse
import torch
import os
import json
from tqdm import tqdm
import math
from aac_metrics import evaluate
from aac_metrics.functional import bleu, meteor, spice,cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
import re
from pycocotools.coco import COCO

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def evaluate_caption(args,predictions:list=[],references:list=[]):
    # results=[]
    # if data == "coco-kapthy":
    #     dt = load_dataset("yerevann/coco-karpathy",split='validation')
    #     reference = dt['sentences']

    # bleu = load("bleu")
    # bleu_results = bleu.compute(predictions=predictions, references=reference)   #prediction: list of strings, reference: list of list
    #scorer = NLGMetricverse(metrics=metrics)
    #scores = scorer(predictions=predictions, references=references,reduce_fn="max")
    #print(f"Current model: {args.model_name} <<>> Max words: {args.max_words}")
    #print("BLEU:")
    b_s, _ = bleu(predictions, references)
    #print(b_s)
    
    #scores = scorer.compute(predictions=predictions, references=references,reduce_fn="max")
    #print("METEOR:")
    m_s, _ = meteor(predictions, references)
    #print(m_s)

    #print("Cider:")
    c_s, _ = cider_d(predictions, references)
    #print(c_s)

    #print("SPICE:")
    s_s, _ = spice(predictions, references)
    #print(s_s)

    return {"BLUE":next(iter(b_s.values())).item(),
            "METEOR": next(iter(m_s.values())).item(),
            "Cider": next(iter(c_s.values())).item(),
            "SPICE":next(iter(s_s.values())).item(),}

def eval_model(args):
    # Model

    answers = [json.loads(q) for q in open(os.path.expanduser(args.answers_file), "r")]
    #answers = json.load(open(os.path.expanduser(args.answers_file), "r"))
    #answers = get_chunk(answers, args.num_chunks, args.chunk_idx)
    if args.annotation:
        if "captions_val2014.json" in args.annotation:
            coco = COCO(os.path.join(args.annotation))
        else:
            anno = json.load(open(os.path.join(args.annotation),'r'))
    if args.QA:
        anno = [json.loads(q) for q in open(os.path.expanduser(args.QA), "r")]
    references=[]        
    predictions=[]
    for i,line in enumerate(tqdm(answers)):
        #print(anno[i]['caption'][0])
        if args.annotation:
            if "captions_val2014.json" in args.annotation:
                caps=[]
                annIds = coco.getAnnIds(imgIds=int(line['image_id']));
                anns = coco.loadAnns(annIds)
                #print(line['image_id'])
                for an in anns:
                    caps.append(an['caption'])

            else:
                caps = [anno[i]['caption']]
        if args.QA:
            caps = [anno[line['question_id']]['label']]
        text=[]
        #print("current caps:",caps)
        for cap in caps:
            if type(cap) == list:
                for c in cap:
                    text.append(pre_caption(c))
            else:
                text.append(pre_caption(cap))
        #print("new caps:",text)
        references.append(text)
        predictions.append(pre_caption(line["text"],max_words=args.max_words))
    print("finish geting data")
    file_score=open(f"./evaluation/{args.model_name}_{args.dataset}.json",'w')
    predictions = preprocess_mono_sents(predictions)
    references = preprocess_mult_sents(references)
    scores=evaluate_caption(args, predictions=predictions,references=references)
    print(scores)
    json.dump(scores, file_score)
    #print(f"bleu@4:{scores['bleu']['score']}, meteor: {scores['meteor']['score']}, CiDer: {scores['cider']['score']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument("--QA", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    args = parser.parse_args()

    eval_model(args)
