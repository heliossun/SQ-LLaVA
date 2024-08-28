import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from peft import PeftConfig, PeftModel
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_Sophon(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base,
                                                                           model_name, device_map='auto')
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    #questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, sources in enumerate(tqdm(questions)):
        id=sources['id']

        question = sources['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        first_answer=sources['conversations'][1]['value']
        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]

        if 'image' in sources:
            image_file = sources['image']
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                fqs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                fqs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            pass
        # First round QA

        conv.append_message(conv.roles[0], fqs)
        conv.append_message(conv.roles[1], first_answer)
        conv.append_message(conv.roles[2], None)
        prompt = conv.get_prompt()
        #print("First QA: ",prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        # first generation is Self-questioning
        questions = []
        answers = []
        for i in range(args.n_shot):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=1.2,
                    top_k=300,
                    top_p=0.95,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=300,
                    use_cache=True)
            # print("out ids: ",output_ids)
            # print("output: ",output_ids[:, input_ids.shape[1]:])
            self_q = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            self_q = self_q.strip()
            if self_q.endswith(stop_str):
                self_q = self_q[:-len(stop_str)]
            self_q = self_q.strip()
            #print(f"VUSER-{i}: ",self_q)
            questions.append(self_q)

            usr_id = tokenizer(conv.roles[1] + ": ").input_ids
            # print(conv.roles[p] + ": ", usr_id)
            usr_id = torch.tensor(usr_id[1:-1], dtype=torch.long, device=output_ids.device)
            usr_id = torch.unsqueeze(usr_id, dim=0)
            # usr_id = usr_id.repeat(output_ids.shape[0], 0)
            curinput_ids = torch.cat((output_ids, usr_id), dim=1)
            with torch.inference_mode():
                output_ids = model.generate(
                    curinput_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
            # print(output_ids)
            outputs = tokenizer.batch_decode(output_ids[:, curinput_ids.shape[1]:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            #print("Assistant: ",outputs)
            answers.append(outputs)
        dialogues=[[questions[0],answers[0]],[questions[1],answers[1]]]
        ans_file.write(json.dumps({"id": id,
                                   "image": image_file,
                                   "sampler": [fqs,first_answer],
                                   "dialogues": dialogues}) + ",\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--question_file", type=str, default="tables/question.json")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--ptlength", type=int, default=10)
    parser.add_argument("--lora_pretrain", type=str, default=None)
    parser.add_argument('--pt', action='store_true')
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--n_shot", type=int, default=5)
    args = parser.parse_args()

    eval_Sophon(args)

