
"""Generate json file for home-office classification."""
import json
import os
import re
import argparse
import ast
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument("--imagefolder", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--input_tyoe", type=str, default="anno")
    args = parser.parse_args()

    if args.input_type == 'raw':
        path = args.imagefolder
        domain = args.domain
        savepath = os.path.expanduser(f'playground/coco')
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
                        "text": "Please describe this image in a few words."
                    }
                    qid+=1
                    json.dump(r, f)
                    f.write('\n')
    else:
        if args.annotation:
            anno = json.load(open(os.path.join(args.annotation),'r'))
        savepath = os.path.expanduser(f'playground/coco')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        qid=0
        with open(savepath+'/question.jsonl', 'w') as f:
            for item in anno: 
                img_pth = item['image']    
                label = item['caption']
                r={
                    "question_id" : qid,
                    "image": img_pth,
                    "text": "Please describe this image in a few words.",
                    "label": label
                }
                qid+=1
                json.dump(r, f)
                f.write('\n')
    