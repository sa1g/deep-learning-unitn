# Test-Time Adaptation (TTA) for image classification 🖼️

In this project, we work on **Test-Time Adaptation (TTA)**, which has recently gained traction due to its ability to enhance model performance without requiring access to training data. It involves improving the robustness of a pre-trained neural network to a test dataset, possibly by improving the network’s predictions on one test sample at a time.

In particular, we focus on **TTA for image classification**, particularly using **CLIP** [[2](#ref-clip2021)] with **TPT** [[3](#ref-tpt2022)]. Our approach involves adapting the model on **single-image test instances**, with the model being reset to its pre-trained state after each instance. This resembles **TTIA**, keeping the constraint of no retention of prior test-time knowledge (between batches, so between images).

<!--- visualize image using html formatting, so that i can scale it properly -->
<p align="center">
  <img src="img/tpt.png" alt="Test-Time Prompt Tuning (TPT) for CLIP" title="Test-Time Prompt Tuning (TPT) for CLIP" width="600" class="center"/><br>
  <em>Figure 1: Test-Time Prompt Tuning (TPT) for CLIP</em>
</p>

Additionally, we experiment with different techiqnues and approaches to try improving either the accuracy or inference speed of the model.


## Setup 🚀

The project is entirely self contained inside the jupyter notebook `report.ipynb`. It can be run using [Google Colab](https://colab.research.google.com/).

## Project Overview 🧠

1. Baseline - ZeroShot CLIP
1. Reproducing TPT: 
   - Reproduce TPT + simplified CoOp (without pretraining) (**Our contribution**)
   - Using OpenAI weights and OpenCLIP implementation
     - Compare zero-shot CLIP OpenAI (weights and implementation) with OpenCLIP (weights and implementation)
   - Using `Kornia` instead of `AugMix` / `torchvision.transforms` (**Extra: Our contribution**)
     - Recreate the AugMix pipeline in Kornia
     - Kornia is faster and can directly run on the GPU
     - Benchmarking the difference
1. Trying to get better at TTA (**Our contribution**)
   - A. Augment Top 10%
   - B. TPT with Top 10%
   - C. Self-Supervised Retrieval (Inspired by DinoV2) [[4](#ref-dinov2)]
   - D. TNT (Recreate the paper)
   - E. TNT with Top 10%
