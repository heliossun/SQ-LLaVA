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

We have also provided a pre-trained weights for the prototype extractor & a two-layer MLP vision-language connector, please download [here](https://huggingface.co/ZachSun/Sophon-projector-cluster-pretrain).

### Visual Instruction Tuning

1. Prepare data
We use two visual instruction-following datset collected by LLaVA.
The single source domain data [llava_instruct_80k.json]() has only COCO image.
The multi source domain data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json),
and the further mixed-up version [Sophon_mix785k.json](https://huggingface.co/datasets/ZachSun/Sophon-780k),
and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Here are the URLs of each dataset, you can use `wget` to download the dataset into a folder.
After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

2. Start training!

Training script with DeepSpeed ZeRO-3 and lora: [`lora_instruct_tuning.sh`](https://github.com/heliossun/Visual-self-QA/lora_instruct_tuning.sh).

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
