# deep-learning-unitn
DL project from the MSc course at UniTN

[Baseline model](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch) taken from NVidia's implementation of The ResNet50 v1.5 model which is a modified version of the original ResNet50 v1 model.

This model was created in 2015. Our objective is to perform TTA on Resnet-A and ImageNet-V2, which were published in 2019 and 2021, so their data was surely not in the original training data.

DL PROJECT

CLIP:

- Base model (CLIP) and dataset loading
- Augment test image
- Initialize prompt

TPT:

- Forward pass with augmented images
- Compute loss
- Confidence selection
- Optimize prompt
- Inference/Classification with optimized prompts

Note: For the choice of parameter group, we optimize the text prompt while keeping the model intact. Our motivation is to avoid distorting the pre-trained features and to preserve the zero-shot generalization ability of pre-trained models.
