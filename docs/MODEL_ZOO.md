# Model Zoo

## SQ-LLaVA-v1.0

| Version | Size | Data | Checkpoint | VQAv2 | GQA | VizWiz | SQA | TextVQA | POPE  | MM-Bench | MM-Bench-CN  | LLaVA-Bench-Wild | MM-Vet |
|----------|----------|-----------|-----------|---|---|---|---|---|---|---|---|---|---|
|SQ-LLaVA | 7B | ShareGPT4V | [ZachSun/sqllava-7B](https://huggingface.co/ZachSun/sqllava-lora-7b) | 80.3 | 63.7 | 55.3 | 70.5 | 60.5 | 87.2  | 66.6 | 60.0  | 74.3 | 37.6 |
| SQ-LLaVA | 13B | ShareGPT4V | [ZachSun/sqllava-13B](https://huggingface.co/ZachSun/sqllava-lora-13) | 81.3 | 65.0 | 58.2 | 71.5 | 61.9 | 87.4  | 68.5 | 62.5 | 80.7 | 39.7 |


<p align="center">
  <img src="../images/2-2.jpg" width="500px"> <br>
  SQ-LLaVA achieves state-of-the-art performance on 9 out of 10 tasks compared with other
open-ended models.
</p>


We provide the LoRA weights (LLM and ViT), please follow the script to load the correct model.

```bash
--model-path ./checkpoints/path/to/ckpt \ 
--model-base Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12_vicuna-7b-v1.5 \ 
--lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \  # ViT LoRA
```
