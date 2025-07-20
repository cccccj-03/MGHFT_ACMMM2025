# <p align="center">MGHFT: Multi-Granularity Hierarchical Fusion Transformer for Cross-Modal Sticker Emotion Recognition, Accepted by ACM MM 2025 </p>

This is the official repository of our ACM Multimedia 2025 Work in Pytorch. Our conference paper is now released at XXXX.

## Abstract
Although pre-trained visual models with text have demonstrated strong capabilities in visual feature extraction, sticker emotion understanding remains challenging due to its reliance on multi-view information, such as background knowledge and stylistic cues. To address this, we propose a novel \textbf{m}ulti-\textbf{g}ranularity \textbf{h}ierarchical \textbf{f}usion \textbf{t}ransformer (\textbf{MGHFT}), with a multi-view sticker interpreter based on Multimodal Large Language Models. Specifically, inspired by the human ability to interpret sticker emotions from multiple views, we first use Multimodal Large Language Models to interpret stickers by providing rich textual context via multi-view descriptions. Then, we design a hierarchical fusion strategy to fuse the textual context into visual understanding, which builds upon a pyramid visual transformer to extract both global and local sticker features at multiple stages. Through contrastive learning and attention mechanisms, textual features are injected at different stages of the visual backbone, enhancing the fusion of global- and local-granularity visual semantics with textual guidance. Finally, we introduce a text-guided fusion attention mechanism to effectively integrate the overall multimodal features, enhancing semantic understanding. Extensive experiments on 2 public sticker emotion datasets demonstrate that MGHFT significantly outperforms existing sticker emotion recognition approaches, achieving higher accuracy and more fine-grained emotion recognition. Compared to the best pre-trained visual models, our MGHFT also obtains an obvious improvement, 5.4\% on F1 and 4.0\% on accuracy. The code is released at https://github.com/cccccj-03/MGHFT\_ACMMM2025.

## Intuition
On the one hand, multimodal foundation models such as LLaVA and GPT-4o excel in image-based generation tasks, leveraging their powerful visual-textual reasoning abilities and extensive background knowledge. On the other hand, vision-language models (VLMs) like CLIP and BLIP, which are pre-trained through image-text alignment, have shown competitive performance in image classification tasks. Robust visual understanding provides a crucial foundation for the SER task, but it alone is not sufficient. More rich information about stickers from multi-views like style and details is also important. The results of the experiment presented in the Figure below also prove this point. As illustrated in the Figure, experimental results on the SER30K dataset reveal a significant performance gap between these state-of-the-art VLMs and our proposed method in the sticker emotion recognition task. This discrepancy highlights an important insight.  Accurately perceiving and interpreting subtle, implicit emotional cues like intention remains a significant challenge. Therefore, achieving effective emotion recognition in stickers requires a more comprehensive integration of contextual knowledge, emotional reasoning, and multi-view understanding.

![image](vlm 8.pdf)

## Innovations:

We design a multi-view sticker interpreter that utilizes MLLMs to decompose stickers into multiple views, providing multi-view descriptions, thereby aligning human perceptual modalities to enrich the understanding of stickers. By introducing the knowledge of MLLMs, our model further improves the performance of emotion recognition.

We design a hierarchical fusion mechanism that injects multi-view textual semantics into different stages of visual feature extraction on the PVT backbone. By leveraging attention and contrastive learning at both global and local granularity, our approach enriches semantic representation and enhances emotion recognition performance.

We propose a novel text-guided multimodal fusion attention mechanism that leverages multi-view textual descriptions to guide the integration of textual and visual modalities, enhancing the model’s understanding of both the contextual knowledge and visual content of stickers.

![image](framework.pdf)

## SER30K dataset

The SER30K dataset should be applied and downloaded at https://github.com/nku-shengzheliu/SER30K.

## MET-MEME dataset

The MET-MEME dataset can be downloaded from https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors.


## Prerequisites

- Python 3.6
- Pytorch 1.10.2
- Others (Pytorch-Bert, etc.) Check requirements.txt for reference.

In addition, please download the ImageNet pre-trained model weights for PVT-small from [PVT](https://github.com/whai362/PVT/tree/v2/classification) and place it in the `./weight` folder.




## Training
To train MGHFT on SER30K on a single node with 2 gpus for 50 epochs run:


```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6666 \
--use_env main.py \
--config configs/pvt/pvt_small.py \
--visfinetune weights/pvt_small.pth \
--output_dir checkpoints/SER \
--dataset SER \
--data-path {path to SER30K dataset} \
--alpha 8 \
--batch-size 16 \
--locals 1 1 1 0
```



## Evaluation
To evaluate MGHFT model performance on SER30K with a single GPU, run the following script using command line:

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=6666 \
--use_env main.py \
--config configs/pvt/pvt_small.py \
--resume checkpoints/SER/checkpoint_best.pth \
--dataset SER \
--data-path {path to SER30K dataset} \
--batch-size 16 \
--alpha 8 \
--locals 1 1 1 0 \
--eval
```

## Citation

If you find this code to be useful for your research, please consider citing. The citation will be released soon.
```shell

```
## Acknowledge
This code is conducted based on [TGCA-PVT](https://github.com/cccccj-03/TGCA-PVT). Thanks for the work of [Ser30K](https://github.com/nku-shengzheliu/SER30K) and [PVT](https://github.com/whai362/PVT). This code is also based on the implementation of them.

