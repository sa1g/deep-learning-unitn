The aim of this report is to provide a comprehensive overview of the background, methodology, sources, thoughts and results of the project. Many information of the **Introduction** come from [[1](#ref-liang2025)]. 

_Even if this is formatted in markdown, references are included._

---

# Introduction

Traditional machine learning models typically assume that training and test data are drawn from the same independent and identically distributed (i.i.d.) distribution. However, this assumption is often violated in real-world scenarios, where the test distribution may differ from the training distribution—a phenomenon known as *distribution shift*. This shift can significantly degrade the generalization performance of machine learning models \[[1](#ref-liang2025)].

To address this challenge, the research community has developed various techniques aimed at improving model robustness under distribution shifts. These methods can be broadly grouped into two main categories:

* **Domain Generalization (DG):** Learning models that generalize well to unseen domains during training.
* **Test-Time Adaptation (TTA):** Adapting a pre-trained model to the test domain during inference without retraining.

In this project, we focus on **Test-Time Adaptation (TTA)**, which has recently gained traction due to its ability to enhance model performance without requiring access to training data.

## Benefits of Test-Time Adaptation

1. **No reliance on training data:** TTA can function without access to the original training data, making it ideal for privacy-sensitive or resource-restricted environments.
2. **Adaptation to unseen domains:** TTA enables models to adapt dynamically to new domains at inference time.
3. **Leverages test data:** By using test data for adaptation, TTA can significantly improve performance on out-of-distribution samples.

   * Unlike DG, which often requires both labeled source and unlabeled target data during training, TTA can be deployed independently, making it more versatile in certain applications.

## Drawbacks of Test-Time Adaptation

1. **Risk of overfitting:** Small or unrepresentative test datasets may lead to overfitting during adaptation.
2. **Assumption of distribution similarity:** TTA generally assumes that test data is not too far from the training distribution, which may not always hold.
3. **Computational overhead:** The additional computation required for adaptation can be prohibitive in low-resource settings.

## Categories of TTA Methods

TTA approaches can be categorized based on the nature and scope of their adaptation strategy:

### 1. **Test-Time Domain Adaptation (TTDA)**

Also known as *source-free domain adaptation*, TTDA uses multiple unlabeled test batches and performs multi-epoch adaptation before making predictions.

### 2. **Test-Time Batch Adaptation (TTBA)**

TTBA adapts the model using individual test batches or instances, with no inter-batch dependency. Each batch is processed independently.

### 3. **Online Test-Time Adaptation (OTTA)**

OTTA adapts the model in an online fashion: test batches are seen only once, and adaptation occurs incrementally using knowledge from previous batches to inform future adaptation.

Let $b_1, \dots, b_m$ represent $m$ unlabeled mini-batches encountered during test time:


- **TTDA**: Uses all $m$ batches for multi-epoch adaptation before final predictions.
- **TTBA**: Adapts on a per-batch or per-instance basis; batches are treated independently.
    - **TTIA**: Also called Episodic Test-Time Adaptation (ETTA), is TTBA with a batch size of 1, where each instance is adapted independently.
- **OTTA**: Adapts sequentially, using prior knowledge from earlier mini-batches to improve performance on subsequent ones.

### Notes on Overlap

* OTTA can be applied in multi-epoch settings similar to TTDA.
* TTBA can be integrated into OTTA frameworks if knowledge from earlier instances is reused.

## Project Scope

In this project, we focus on **TTA for image classification**, particularly using **CLIP** [[2](#ref-clip2021)] with **TPT** [[3](#ref-tpt2022)]. Our approach involves adapting the model on **single-image test instances**, with the model being reset to its pre-trained state after each instance. This resembles **TTIA**, keeping the constraint of no retention of prior test-time knowledge (between batches, so between images).

TODO: add Fig. 1 (a) from the paper [recreate it]

A comprehensive list of TTA methods is provided at [GitHub - TTIA, Clip Related](https://github.com/tim-learn/awesome-test-time-adaptation/blob/main/TTA-TTBA.md/#clip-related).


# TTIA
> **Definition**: "_Test-Time Instance Adaption, TTIA_ Given a classifier $f_\mathcal{S}$ learned on the source domain $\mathcal{D_s}$, and an unlabeled target instance $x_t \in \mathcal{D_T}$ under distribution shift, _test-time instance adaption_ aims to leverage the labeled knowledge implied in $\mathcal{f_S}$ to infer the label of $x_t$ adaptively" [[1](#ref-liang2025)]. In other words, TTIA aims to adapt the classifier $f_\mathcal{S}$ to the target instance $x_t$ by leveraging the knowledge of the source domain $\mathcal{D_S}$.

TTIA differs from TTBA in that single-instance adaption is performed, instead of batch-wise adaption, giving an example the difference is between classifying a single frame of a video and classifying a sequence of frames. In both methods no memory of the previous test-time knowledge is retained.

## Common Techniques (and some not only for TTIA)
It follows a list of techniques that can be used for this project (TTIA for image classification). This list is not exhaustive, a bigger one can be found in the paper [[1](#ref-liang2025)] and linked above in the **Project Scope** section. 
This list has the objective to provide a quick overview of the techniques that can be used for TTIA, and to give some ideas for possible improvements on TPT [[3](#ref-tpt2022)].

### Test-Time Augmentation 
Generally, test-time augmentation techniques do not explicitly consider distribution shifts but can be advantageous for TTA methods.
Most known are:
- AugMix
- estimate uncertainty: Smith, L., & Gal, Y. (2018). Understanding measures of uncertainty  for adversarial example detection. In Proceedings of UAI (pp. 560–569).
- enhance robustness: Pérez, J. C., Alfarra, M., Jeanneret, G., Rueda, L., Thabet, A., Ghanem, B., & Arbeláez, P. (2021). Enhancing adversarial robustness via test-time transformation ensembling. In Proceedings of ICCV (pp. 81–91).
- we could even try to get inspired by **MobileCLIP**'s reinforcement dataset method.

### Batch Normalization Calibration
Batch normalization (BN) is a widely used technique in deep learning that normalizes the activations of each layer in a mini-batch. This is done to improve the training speed and stability of the model. Obviously, methods that rectify BN suffer when the batch size is small and many are not directly applicable to TTIA, as they requir multiple test batches. However, some techniques have been proposed to adapt BN statistics during test time:
- SaN [[4](#ref-san2022)] attempts to mix instance normalizatoin (IN [[5](#ref-in2016)]) statistics estimated per instance with the training BN statistics.
- InstCal [[6](#ref-instcal2022)]  introduces a module to learn the interpolating weight between (IN [[5](#ref-in2016)]) and BN statistics, allowing the network to adjust the importance of training statistics for each test instance. > TODO: check this, it looks like this paper has nothing to do with TTIA.
- SITA [[7](#ref-sita2021)] expands a single instance to a bathc of instance using random augmentation, then estimates the BN using the weighted average over these augmented instances.

### Model Optimization
Involces adjusting the parameters of a pre-trained model for each unlabled test instance. There are two main approaches:
1. **Training with auxiliary tasks**: introduce a self-supervised task in the primary during both training and test phases
    - **Not** directly **applicable** to TTIA and our project, as it requires not only TTA but also training.
1. **Fine-tuning**: designing task-specific opjective for updating the pre-trained model.

#### Training-Agnostic Fine-Tuning
Fine-tuning with unsupervised objectives.
- MEMO [[8](#ref-memo2021)]: optimizes the netropy of the averagd prediction over multiple random aumgentations of the input sample.
- PromptAlign [[9](#ref-promptalign2023)]: handle train-test distribution shift by aligning mean and variances fo the test sample and the source dataset  statistics. TODO: check if it's compatible.
- SUTA [[10](#ref-suta2022)]: minimum class confusion to reduce uncertainty. This is for Automatic Speech Recognition, but maybe something can be used for image classification. TODO: check if it's compatible.
- RLCF [[11](#ref-rl2023)]: TTA with feedback to rectify the model output and prevent the model from becoming blindly confident - reinforcement learning with CLIP feedback.
- Deep Matching [[12](#ref-deepmatching)]: image matching with a pre-trained model. TODO: check compatibility.
- Image generation, which Ettore doesn't like so he didn't search any paper about it.

### Meta-Learning
Not directly applicable to our project (TODO: check if I'm wrong, 99% i'm right). Won't be analyzed further.

### Input Adaption
Focuses on changing input data for pre-trained models. Notable techniques include:
- TPT [[3](#ref-tpt2022)]: freezes the pre-trained multimodal model (CLIP) and only learns the extra text prompt based on the marginal entropy of each instance.
- CVP [[13](#ref-cvp2023)]: optimizes the convolutional visual prompts in the input under the guidance of a self-supervised contrastive learning objective. TODO: check compatibility.
- TTA-AE, TTA-DAE, TTO-AE, AdvTTT may be interesting, skipped.
- TAF-Cal [[14](#ref-tafcal2022)]: utilizes the average amplitude feature over the training data to perform Fourier style calibration at both training and test phases. TODO: check compatibility.
- data manifold constraint can aid in achieving better alignemnt between test data and unseen training data.
    - probably not compatible, it's in the survey.
- ESA [[15](#ref-esa2023)]: update the target feature by energy minimization through Langevin dynamics. TODO: check compatibility.
- AntiAdv [[16](#ref-antiadv)]: anti-adversary layer, aimed at countering this effect. In particular, our layer generates an input perturbation in the opposite direction of the adversarial one and feeds the classifier a perturbed version of the input. Our approach is training-free and theoretically supported

### Dynamic Inference
- LAME [[17](#ref-lame2022)]: neighbor conssitency to enforce consistent assignments on neighboring points in the feature space, without modifying the pre-trained mdoel.

### Other (not born for TTA/TTIA)
- **Label Propagation for Deep Semi-supervised Lear**ning: which seeks local smoothness by maximizing the pairwise similarities between nearby data points. For TTA tasks, these semi-supervised learning techniques can be easily integrated to unsupervisedly update the pre-trained model during adaptation.

---

# Model and Dataset
The base model is VitB/32. Dataset is ImageNetA.

# Methodology
Introduce with which logic everything is done, and the logic behind the choices made.


## TPT
how it works, and results
## TPT + CoCo
results

## Our test-time adaptation method
ideas, thoughts, expected results and effective results

---

# Implementations


---

# Bibliography
1. <a id="ref-liang2025"></a> Liang J., He R., Tan T. (2025). _A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts_. IJCV, 133, 31-64, [Link](https://doi.org/10.1007/s11263-024-02181-w)
1. <a id="ref-clip2021"></a> Radford A., Kim K. J., Hallacy C., et al. (2021). _Learning Transferable Visual Models From Natural Language Supervision_. ICML 2021, 8748-8763, [Link](https://arxiv.org/abs/2103.00020)
1. <a id="ref-tpt2022"></a> Shu M., Nie W., Huang D., et al. (2022). _Test-time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models_. NeurIPS 2022, [Link](https://arxiv.org/abs/2209.07511)
1. <a id="ref-san2022"></a> Bahmani S., et al. (2022) _Semantic Self-adaptation: Enhancing Generalization with a Single Sample_. TMLR 2023, [Link](https://arxiv.org/abs/2208.05788v2)
1. <a id="ref-in2016"></a> Ulyanov D., et al. (2016). _Instance Normalization: The Missing Ingredient for Fast Stylization_. [Link](https://arxiv.org/abs/1607.08022)
1. <a id="ref-instcal2022"></a> Zou. Y., et al. (2022). _PseudoSeg: Designing Pseudo Labels for Semantic Segmentation_. ICLR 2021, [Link](https://arxiv.org/abs/2010.09713)
1. <a id="ref-sita2021"></a> Khurana A., et al. (2021). _SITA: Single Image Test-time Adaptation_. [Link](https://arxiv.org/abs/2112.02355)
1. <a id="ref-memo2021"></a> Zhang M., et al., (2021). _MEMO: Test Time Robustness via Adaptation and Augmentation_, NeurIPS 2022, [Link](https://arxiv.org/abs/2110.09506)
1. <a id="ref-promptalign2023"></a> Hassan J., et al. (2023). _Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization_. NeurIPS 2023, [Link](https://arxiv.org/abs/2311.01459)
1. <a id="ref-suta2022"></a> Liu, G. at al. (2022). _Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition_. Interspeech 2022, [Link](https://arxiv.org/abs/2203.14222)
1. <a id="ref-rl2023"></a> Zhao S., at al. (2023). _Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models_. ICLR 2024, [Link](https://arxiv.org/abs/2305.18010)
12. <a id="ref-deepmatching"></a> Hong S., Kim S. (2021) _Deep Matching Prior: Test-Time Optimization for Dense Correspondence_. IEEE/CVF 2021, [Link](https://arxiv.org/abs/2106.03090)
1. <a id="ref-cvp2023"></a> tsai Y., et al. (2023). _Convolutional Visual Prompt for Robust Visual Perception_, [Link](https://arxiv.org/abs/2303.00198)
1. <a id="ref-tafcal2022"></a> Zhao X., et al. (2022). _Test-time Fourier Style Calibration for Domain Generalization_. IJCAI 2022, [Link](https://arxiv.org/abs/2205.06427)
1. <a id="ref-esa2023"></a> Xiao Z., et al. (2023). _Energy-Based Test Sample Adaptation for Domain Generalization_. ICLR 2023, [Link](https://arxiv.org/abs/2302.11215)
1. <a id="ref-antiadv"></a> Alfarra M., at al. (2021). _Combating Adversaries with Anti-Adversaries_. AAAI 2022, [Link](https://arxiv.org/abs/2103.14347)
1. <a id="ref-lame2022"></a> Boudiaf M., et al. (2022). _Parameter-free Online Test-time Adaptation_. CVPR 2022. [Link](https://arxiv.org/abs/2201.05718)



## Unused 
Ishii, M., & Sugiyama, M. (2021). Source-free domain adaptation via
distributional alignment by matching batch normalization statis-
tics. arXiv:2101.10842.

Iwasawa, Y., & Matsuo, Y. (2021). Test-time classifier adjustment mod-
ule for model-agnostic domain generalization. In Proceedings of
NeurIPS (pp. 2427–2440).

Jain, V., & Learned-Miller, E. (2011). Online domain adaptation of a
pre-trained cascade of classifiers. In Proceedings of CVPR (pp.
577–584).