import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from sqllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from sqllava.conversation import conv_templates, SeparatorStyle
from sqllava.model.builder import load_pretrained_model
from sqllava.utils import disable_torch_init
from sqllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from peft import PeftConfig, PeftModel
from PIL import Image
import math




def eval_sqllava(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base,
                                                                           model_name, device_map='auto')
    image_path="002.jpg"
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print("couldn't find image")
        
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    fqs = DEFAULT_IMAGE_TOKEN + '\n' + "Can you describe the image in detail?"
    #First round QA
    conv.append_message(conv.roles[0], fqs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    first_answer = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    first_answer = first_answer.strip()
    if first_answer.endswith(stop_str):
        first_answer = first_answer[:-len(stop_str)]
    first_answer = first_answer.strip()
    conv.clear_message()
    conv.append_message(conv.roles[0], fqs)
    conv.append_message(conv.roles[1], first_answer)
    conv.append_message(conv.roles[2], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
        0).cuda()
    # Visual self-questioning
    for i in range(args.n_shot):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=1.2,
                top_k=300,
                top_p=0.95,
                # no_repeat_ngram_size=3,
                max_new_tokens=300,
                use_cache=True)
        self_q = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        self_q = self_q.strip()
        if self_q.endswith(stop_str):
            self_q = self_q[:-len(stop_str)]
        self_q = self_q.strip()
        print(f"VUSER:{i}: ",self_q)

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--lora_pretrain", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--n_shot", type=int, default=5)
    args = parser.parse_args()

    eval_sqllava(args)
    
