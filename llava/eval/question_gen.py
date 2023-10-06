
"""Generate json file for home-office classification."""
import json
import os
import re
import argparse
import ast
import pickle

from pycocotools.coco import COCO

   
def get_all_data(data_root: str, suffix: str):
    data = []
    for i in range(16):
        out_data_path = f"{data_root}/conceptual_{suffix}_{i:02d}.pkl"
        if os.path.isfile(out_data_path):
            with open(out_data_path, 'rb') as f:
                raw_data = pickle.load(f)["info"]
            data.append(raw_data)

    return data


def collect(data_root: str, suffix: str):
    raw_data = get_all_data(data_root, suffix)
    data = []
    for thread_data in raw_data:
        for item in thread_data:
            data.append((item, thread_data[item]["caption"]))
    return data


def getitem(data_root,suffix):
    return collect(data_root, suffix)
    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument("--imagefolder", type=str, default="")
    parser.add_argument("--answers_file", type=str, default=None)
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--input_type", type=str, default="anno")
    args = parser.parse_args()

    if args.input_type == 'raw':
        path = args.imagefolder
        domain = args.domain
        savepath = os.path.expanduser(f'playground/coco2014')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        qid=0
        with open(savepath+'/question.jsonl', 'w') as f:
            for cls_name in os.listdir(os.path.join(path,domain)):
                pth = os.path.join(path,domain,cls_name)
                for image in os.listdir(pth):
                    img_pth = os.path.join(domain,cls_name,image)
                    r={
                        "question_id" : qid,
                        "image": img_pth,
                        "text": "Provide a brief description of the given image, your answer should be in one sentence."
                    }
                    qid+=1
                    json.dump(r, f)
                    f.write('\n')
    elif args.input_type =='pickle':
        savepath = os.path.expanduser(f'playground/conceptual')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        data_root=args.imagefolder
        suffix='val'
        data=getitem(data_root,suffix)
        qid=0
        with open(savepath+'/question3.jsonl', 'w') as f:
            for item in data:
                image_name, caption = item
                image_path = f"{suffix}/{image_name}.jpg"
                r={
                    "question_id" : qid,
                    "image": image_path,
                    "text": "Provide a brief description of the given image, your answer should be in one sentence.",
                    "label": caption
                }
                qid+=1
                json.dump(r, f)
                f.write('\n')
    else:
        if args.annotation:
            if "captions_val2014.json" in args.annotation:
                coco = COCO(os.path.join(args.annotation))
                #ids = list(coco.anns.keys())
            else:
                anno = json.load(open(os.path.join(args.annotation),'r'))
        savepath = os.path.expanduser(f'playground/coco2014')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        qid=0
        with open(savepath+'/question.jsonl', 'w') as f:
            if "captions_val2014.json" in args.annotation:
                baseline_answers = json.load(open(os.path.expanduser(args.answers_file), "r"))
                for i,line in enumerate(baseline_answers):
                    ImgId = line['image_id']
                    ann = coco.loadImgs(int(ImgId))[0]
                    img_id=ann['id']
                    img_pth = ann['file_name']
                    r={
                        "question_id" : qid,
                        "image": img_pth,
                        "text": "Provide a brief description of the given image, your answer should be in one sentence.",
                        "image_id": img_id,
                    }
                    qid+=1
                    json.dump(r, f)
                    f.write('\n')
            else:
                for item in anno: 
                    img_pth = item['image']    
                    label = item['caption']
                    r={
                        "question_id" : qid,
                        "image": img_pth,
                        "text": "Provide a brief description of the given image, your answer should be in one sentence.",
                        "label": label
                    }
                    qid+=1
                    json.dump(r, f)
                    f.write('\n')
    