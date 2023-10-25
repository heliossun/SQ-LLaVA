import json
import argparse
import os
import urllib.request
def download_nocaps(args):
    anno = json.load(open(os.path.join(args.anno),'r'))
    for img in anno["images"]:
        #print(img)
        if img["domain"] != "in-domain":
            d=img['domain']
            fn=img['file_name']
            urllib.request.urlretrieve(img["coco_url"],f"{args.save}/images/{d}/{fn}")
def generate_annos(args):
    anno = json.load(open(os.path.join(args.anno),'r'))
    nd=[]
    od=[]
    texts=[[] for i in range(len(anno["images"]))]
    for t in anno["annotations"]:
        texts[t["image_id"]].append(t['caption'])
    with open('/home/gs4288/LLaVA/playground/nocaps/near-domain_question.jsonl', 'w') as f:
        qid=0
        for i in anno["images"]:
            if i["domain"] == "near-domain":
                nd.append({'image': i["file_name"],
                          'caption': texts[i['id']]})
                r={
                "question_id": qid,
                "image": i["file_name"],
                "text": "Provide a brief description of the given image, your answer should be in one sentence."
            }
                qid += 1
                json.dump(r, f)
                f.write('\n')

    with open('/home/gs4288/LLaVA/playground/nocaps/out-domain_question.jsonl', 'w') as f:
        qid=0
        for i in anno["images"]:
            if i["domain"] == "out-domain":
                od.append({'image': i["file_name"],
                           'caption': texts[i['id']], })
                r={
                    "question_id": qid,
                    "image": i["file_name"],
                    "text": "Provide a brief description of the given image, your answer should be in one sentence."
                }
                qid += 1
                json.dump(r, f)
                f.write('\n')
    with open("/home/gs4288/data/nocaps/nocaps_val_near_domain.json", "w") as final:
        json.dump(nd, final)
    with open("/home/gs4288/data/nocaps/nocaps_val_out_domain.json", "w") as final:
        json.dump(od, final)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", type=str, default="/home/gs4288/data/nocaps/nocaps_val_4500_captions.json")
    parser.add_argument("--save", type=str, default="/home/gs4288/data/nocaps")

    args = parser.parse_args()

    #download_nocaps(args)
    generate_annos(args)