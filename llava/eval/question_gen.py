
"""Generate json file for home-office classification."""
import json
import os
import re
import argparse
import ast
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagefolder", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    args = parser.parse_args()

    path = args.imagefolder
    domain = args.domain
    savepath = os.path.expanduser(f'playground/office-home/{domain}')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    qid=0
    with open(savepath+'/question2.jsonl', 'w') as f:
        for cls_name in os.listdir(os.path.join(path,domain)):
            pth = os.path.join(path,domain,cls_name)
            for image in os.listdir(pth):
                img_pth = os.path.join(domain,cls_name,image)
                r={
                    "question_id" : qid,
                    "image": img_pth,
                    "text": "Please find out what is the main focus of the image? Your answer should be in one word.",
                    "label": cls_name
                }
                qid+=1
                json.dump(r, f)
                f.write('\n')