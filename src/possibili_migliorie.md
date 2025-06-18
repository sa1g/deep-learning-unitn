# PROMPT ENCODER
## 9. Prompt Align
Cool, **skip**. 
Usa un dataset proxy per allineare i prompt, quindi e' sensibile a questo dataset. Difficile da usare in TTA.

## 13. Convolutional Visual Prompt for Robust Visual Perception
Cool, **skip**
We improve robustness at test time by dynamically
adapting to unforeseen corruptions and unknown shifts.

Add a convolutional visual prompt to the input image to get better insights and improve performances 
of the prompt.

Figo assai, ma bisogna fare training. Non si puo' usare.

## 15. ENERGY-BASED TEST SAMPLE ADAPTATION FOR DOMAIN GENERALIZATION

# IMAGE ENCODER
## 7. SITA: Single Image TTA
we focus on the challenging SITA setting, where
inference has to be done on a single image at a time, and not
a batch of instances, with the model being reset to the source
model after every test instance adaptation.

1. image augment
2. filter image by entropy (optional)
3. calculate batch norm statistics (mean, std)
4. substitute batch norm statistics with the calculated ones



### Idea - [Link](https://chat.deepseek.com/a/chat/s/fc66a5b4-3567-4772-aab8-c2b86788c712)

SITA works on batch norm. CLIP has only layer norm. By playing with
TPT we already have the augmented vjews and a wa to keep the _best_ (entropy). The idea is to:
1. calculate per-instance statistics ($\sigma, \mu$) using the best augmented view
2. aggregate: average
3. use adapted statistics  
    Two approaches:
    - reuse CLIP original statistics normalized by the new mean and std
    - scale dinamically: $\gamma* = \gamma * \sigma, \beta* = \beta * \mu$
    - use the new statistics as is


# Bibliography

6. FORSE INUTILE (non visto) <a id="ref-instcal2022"></a> Zou. Y., et al. (2022). _PseudoSeg: Designing Pseudo Labels for Semantic Segmentation_. ICLR 2021, [Link](https://arxiv.org/abs/2010.09713)
7. <a id="ref-sita2021"></a> Khurana A., et al. (2021). _SITA: Single Image Test-time Adaptation_. [Link](https://arxiv.org/abs/2112.02355)
9. (vedere compatibilita') <a id="ref-promptalign2023"></a> Hassan J., et al. (2023). _Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization_. NeurIPS 2023, [Link](https://arxiv.org/abs/2311.01459)
10. (vedere compatibilita') <a id="ref-suta2022"></a> Liu, G. at al. (2022). _Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition_. Interspeech 2022, [Link](https://arxiv.org/abs/2203.14222)
11. <a id="ref-rl2023"></a> Zhao S., at al. (2023). _Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models_. ICLR 2024, [Link](https://arxiv.org/abs/2305.18010)
13. (vedere compatibilita') <a id="ref-cvp2023"></a> tsai Y., et al. (2023). _Convolutional Visual Prompt for Robust Visual Perception_, [Link](https://arxiv.org/abs/2303.00198)
14. (vedere compatibilita') <a id="ref-tafcal2022"></a> Zhao X., et al. (2022). _Test-time Fourier Style Calibration for Domain Generalization_. IJCAI 2022, [Link](https://arxiv.org/abs/2205.06427)
15. (vedere compatibilita') <a id="ref-esa2023"></a> Xiao Z., et al. (2023). _Energy-Based Test Sample Adaptation for Domain Generalization_. ICLR 2023, [Link](https://arxiv.org/abs/2302.11215)

new: https://arxiv.org/abs/2502.06855