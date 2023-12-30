# Please follow GitHub repo to organize the data 

# 1. Download json files:
mkdir mixTraindata
# run the following command inside ./mixTraindata
git lfs clone "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
git lfs clone "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V"



# 2. Download images
mkdir mixTraindata/llava/llava_pretrain
wget -P mixTraindata/llava/llava_pretrain https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip 

mkdir mixTraindata/coco
wget -P mixTraindata/coco http://images.cocodataset.org/zips/train2017.zip

mkdir mixTraindata/gqa
wget -P mixTraindata/gqa https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

mkdir mixTraindata/ocr_vqa
# save "dataset.json" from https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_ into "mixTraindata/ocr_vqa"
# put the file under "./ocr_vqa" into "mixTraindata/ocr_vqa" and run "python loadDataset.py"

mkdir mixTraindata/textvqa
wget -P mixTraindata/textvqa https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

mkdir mixTraindata/vg
wget -P mixTraindata/vg https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -P mixTraindata/vg https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip



# Down share_textvqa, web-celebrity, web-landmark, wikiart from:
# https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing
# and put into the following folders

mkdir mixTraindata/share_textvqa/images
mkdir mixTraindata/web-celebrity/images
mkdir mixTraindata/web-landmark/images
mkdir mixTraindata/wikiart/images


mkdir mixTraindata/sam
# run "python download_sam.py --processes 4 --input_file sa1b_links.txt --raw_dir mixTraindata/sam/raw --images_dir mixTraindata/sam/images --masks_dir mixTraindata/sam/annotations" 
# to download sam images