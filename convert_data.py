import argparse
import os
import json
import math

def convert_to_AQ(args):
    # Model
    
    data = json.load(open(args.data_path, "r"))
    
    output_file = os.path.expanduser(args.out_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_file = open(output_file, "w")
    
    for d in data:
        holder=None
        convs=[]
        for i, e in enumerate(d['conversations']):
            from_=e['from']
            value=e['value']
            if from_ == "human":
                value.replace("\n<image>","")
                value.replace("<image>\n","")
                holder={"from":'gpt',
                            "value":value}
            elif from_ == "gpt":
                if i==1:
                    value = "<image>"+'\\n'+value
                    #print(value)
                convs.append({"from":'human',
                            "value":value
                            })
                convs.append(holder)
            
        out_file.write(json.dumps({"id": d['id'],
                                   "image": d['image'],
                                   "conversations": convs,}) + ",\n")
        out_file.flush()
    
    out_file.close()

def convert_to_QAQA(args):
    # Model
    
    data = json.load(open(args.data_path, "r"))
    
    output_file = os.path.expanduser(args.out_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_file = open(output_file, "w")
    
    for d in data:
        holder=None
        convs=[]
        for i, e in enumerate(d['conversations']):
            from_=e['from']
            value=e['value']
            if from_ == "human":
                value.replace("\n<image>","")
                value.replace("<image>\n","")
                holder={"from":'gpt',
                            "value":value}
            elif from_ == "gpt":
                if i==1:
                    value = "<image>"+'\\n'+value
                    #print(value)
                convs.append({"from":'human',
                            "value":value
                            })
                convs.append(holder)
            
        out_file.write(json.dumps({"id": d['id'],
                                   "image": d['image'],
                                   "conversations": convs,}) + ",\n")
        out_file.flush()
    
    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./llava_instruct_80k.json")
    parser.add_argument("--out_file", type=str, default="./llava_instruct_aq_80k.json")
    args = parser.parse_args()
    
    convert_to_AQ(args)

