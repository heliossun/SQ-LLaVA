# Self-questioning for Vision-Language Assistant

- [x] Self-questioning
- [x] Cross-attention as vision projector
- [x] Data augmentation
- [x] Light-weight
- [x] Prototype extractor


## Contents
- [Install](#install)
- [LLaVA Weights](#llava-weights)
- [Demo](#Demo)
- [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- [Train](#train)
- [Evaluation](#evaluation)


1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```




## Train
Training consists of two stages: (1) feature alignment stage: use our 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5-7B | 256 | 1e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5-7B | 128 | 2e-5 | 1 | 2048 | 0 |



### Download Vicuna checkpoints (automatically)

Our base model Vicuna v1.5, which is an instruction-tuned chatbot, will be downloaded automatically when you run our provided training scripts. No action is needed.

### Pretrain (feature alignment)

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/heliossun/Visual-self-QA/edit/main/pretrain.sh).

- `--mm_projector_type cluster`: the prototype extractor & a two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

We have also provided a pre-trained weights for the prototype extractor & a two-layer MLP vision-language connector, please download [here](https://huggingface.co/ZachSun/Sophon-projector-cluster-pretrain) and put in `./checkpoints/projector`.

### Visual Instruction Tuning

1. Prepare data
## Data

| Data file name | Size |
| --- | ---: |
| [sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json) | 134 MB |
| [share-captioner_coco_lcs_sam_1246k_1107.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json) | 1.5 GB |
| [sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json) | 1.2 GB |

### ShareGPT4V Dataset
This dataset is curated from LAION, CC, SBU, SAM, COCO, web-landmark, web-celebrity, wikiart, etc, resulting in total 102K high-quality image-text pairs with the help of powerful GPT4-Vision.

### ShareGPT4V-PT Dataset
The pretraining dataset used in this release is a mixture of LAION, CC, SBU, SAM, COCO datasets, resulting in total 1246K image-text pairs with the help of our general ShareCaptioner

### SFT Dataset
We replace 23K image-text pairs related to the image captioning task in LLaVA-mix-665K with a equivalent subset in our collected GPT4V-generated high-quality image-text pairs.

### Prepare Images

First, download all images we used.

- LAION-CC-SBU-558K: [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- WebData: [images](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Only for academic usage.
- SAM: [images](https://ai.meta.com/datasets/segment-anything-downloads/). We only use 000000~000050.tar for now.
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- TextVQA: [trainvalimages](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Then, organize the data as follows in `./mixTraindata`:

```none
Visual-self-qa
├── ...
├── mixTraindata
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
│   ├── gqa
│   │   ├── images
│   ├── ocr_vqa
│   │   ├── images
│   ├── textvqa
│   │   ├── train_images
│   ├── vg
│   │   ├── VG_100K
│   │   ├── VG_100K_2
│   ├── share_textvqa
│   │   ├── images
│   ├── web-celebrity
│   │   ├── images
│   ├── web-landmark
│   │   ├── images
│   ├── wikiart
│   │   ├── images
│   ├── share-captioner_coco_lcs_sam_1246k_1107.json
│   ├── sharegpt4v_instruct_gpt4-vision_cap100k.json
│   ├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
├── ...
```

**Important notice**: For the convenience, we provide a zip file for web data. These images must be used for academic purpose.


2. Start training!

Training script with DeepSpeed ZeRO-3 and lora: [`lora_instruct_tuning.sh`]([https://github.com/heliossun/Visual-self-QA/lora_instruct_tuning.sh](https://github.com/heliossun/Visual-self-QA/blob/main/lora_instruct_tuning.sh)).

- `--mm_projector_type cluster`: the prototype extractor & a two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--group_by_modality_length True`: this should only be used when your instruction tuning dataset contains both language (e.g. ShareGPT) and multimodal (e.g. LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.

3. Training with self-questioning
- `--version v1_sq`. 

## Evaluation
Prepare data
Please download raw images of datasets (COCO, Flickr, nocaps, conceptual, concadia) for image captioning tasks.
1. Evaluate models on image captioning.
See [experiments.sh/experiments2.sh](https://github.com/heliossun/Visual-self-QA/edit/main/experiments.sh) on 7 datasets.
2. Evaluate models on a diverse set of 12 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

3. To test self-questioning
- `--version v1_sq`: use the self-questioning templage.
- `--sq`: enable the self-questioning function.
- `--n_shot 3`: the number of generated questions. 

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.

## Related Projects
