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
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_Sophon(args):
    sq_prompt = [1]
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base,
                                                                           model_name, device_map='auto')
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        #keywords = [stop_str]

        image_path = os.path.join(args.image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print("couldn't find image")
            continue
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        if model.config.mm_use_im_start_end:
            fqs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:

            fqs = DEFAULT_IMAGE_TOKEN + '\n' + "Can you describe the image in detail?"
        #First round QA
        conv.append_message(conv.roles[0], fqs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        #stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
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
        #print("First round QA: ",fqs, first_answer)

        #self questioning
        fqs = DEFAULT_IMAGE_TOKEN + '\n'
        conv.clear_message()
        #conv.append_message(conv.roles[0], fqs)
        #conv.append_message(conv.roles[1], first_answer)
        conv.append_message(conv.roles[2], fqs)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        #print("QA + Self Q: ",prompt)
        # print("input ids: ", tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'))
        # first generation is Self-questioning
        questions = []
        answers = []
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
            # print("out ids: ",output_ids)
            # print("output: ",output_ids[:, input_ids.shape[1]:])
            self_q = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            self_q = self_q.strip()
            if self_q.endswith(stop_str):
                self_q = self_q[:-len(stop_str)]
            self_q = self_q.strip()
            #print(f"VUSER-{i}: ",self_q)
            questions.append(self_q)
            for p in sq_prompt:
                usr_id = tokenizer(conv.roles[p] + ": ").input_ids
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
                answers.append(outputs)
                # print(f"ASSISTANT-{i}: {outputs}")

        conv.clear_message()
        conv.append_message(conv.roles[0], fqs)
        conv.append_message(conv.roles[1], first_answer)
        for i in range(len(questions)):
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print("<<<<<>>>>>",prompt)
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
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs)
        # predictions.append(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

def eval_sq(args):

    sq_prompt=[1]
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base,
                                                                           model_name, device_map='auto')
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        #stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        image_path = os.path.join(args.image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print("couldn't find image")
            continue
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        self_q = ''
        if model.config.mm_use_im_start_end:
            self_q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
        else:
            self_q = DEFAULT_IMAGE_TOKEN + '\n'
        conv.append_message(conv.roles[2], self_q)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[:-2].unsqueeze(
                0).cuda()
        #print(prompt)
        #print("input ids: ", tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'))
        # first generation is Self-questioning
        questions=[]
        answers=[]
        for i in range(args.n_shot):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    top_k=50,
                    top_p=0.85,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=200,
                    use_cache=True)
            #print("out ids: ",output_ids)
            #print("output: ",output_ids[:, input_ids.shape[1]:])
            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            #print(f"VUSER-{i}: ",outputs)
            questions.append(outputs)
            for p in sq_prompt:
                usr_id = tokenizer(conv.roles[p] + ": ").input_ids
                #print(conv.roles[p] + ": ", usr_id)
                usr_id = torch.tensor(usr_id[1:-1], dtype=torch.long, device=output_ids.device)
                usr_id = torch.unsqueeze(usr_id, dim=0)
                #usr_id = usr_id.repeat(output_ids.shape[0], 0)
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
                #print(output_ids)
                outputs = tokenizer.batch_decode(output_ids[:, curinput_ids.shape[1]:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                answers.append(outputs)
                #print(f"ASSISTANT-{i}: {outputs}")

        conv.clear_message()
        qs1 = DEFAULT_IMAGE_TOKEN + '\n' + questions[0]
        conv.append_message(conv.roles[0], qs1)
        conv.append_message(conv.roles[1], answers[0])
        for i in range(1, len(questions)):
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1],answers[i])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


        # qa_prompt = conv.roles[0] + ": " + qs + " "
        # qa_prompt += conv.roles[1] + ": "
        # current_ids = tokenizer(qa_prompt).input_ids
        # #print("regular QA prompt: ",qa_prompt)
        # #print("<<<>>>>>>: ",current_ids)
        # #print("final QA ids:", current_ids)
        # current_ids = torch.tensor(current_ids[1:-1], dtype=torch.long, device=output_ids.device).unsqueeze(0)
        #
        # input_ids = torch.cat((output_ids, current_ids), dim=1)
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
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print("Final ASSISTANT: ",outputs)
        # predictions.append(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, 
                                                                           model_name,device_map='auto',lora_pt=args.lora_pretrain)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_path = os.path.join(args.image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print("couldn't find image")
            continue
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        #print("stop str: ",stop_str)
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if (conv.version == "v0" or conv.version == "dolly") else None
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
                stopping_criteria=stopping_criteria,
                use_cache=True)
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        #print("output",outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--ptlength", type=int, default=10)
    parser.add_argument("--lora_pretrain", type=str, default=None)
    parser.add_argument('--pt', action='store_true')
    parser.add_argument('--pt_v', action='store_true')
    parser.add_argument("--sq", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--n_shot", type=int, default=5)
    args = parser.parse_args()

    if args.sq:
        eval_Sophon(args)
    else:
        eval_model(args)
