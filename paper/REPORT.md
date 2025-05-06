The aim of this report is to provide a comprehensive overview of the background, methodology, sources, thoughts and results of the project. Many information of the **Introduction** come from [[1](#ref-liang2025)]. 

_Even if this is formatted in markdown, sources are included._

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
    - **TTIA**: TTBA with a batch size of 1, where each instance is adapted independently.
- **OTTA**: Adapts sequentially, using prior knowledge from earlier mini-batches to improve performance on subsequent ones.

### Notes on Overlap

* OTTA can be applied in multi-epoch settings similar to TTDA.
* TTBA can be integrated into OTTA frameworks if knowledge from earlier instances is reused.

## Project Scope

In this project, we focus on **TTA for image classification**, particularly using **CLIP** [[2](#ref-clip2021)] with **TPT** [[3](#ref-tpt2022)]. Our approach involves adapting the model on **single-image test instances**, with the model being reset to its pre-trained state after each instance. This resembles **TTIA**, keeping the constraint of no retention of prior test-time knowledge (between batches, so between images).

TODO: add Fig. 1 (a) from the paper [recreate it]

A comprehensive list of TTA methods is provided at [GitHub - TTIA](https://github.com/tim-learn/awesome-test-time-adaptation/blob/main/TTA-TTBA.md/#Instance-level).

# TTIA

## Definition

## Common Ideas

# Model and Dataset

# Methodology

## TPT
implementation and results
## TPT + CoCo
implementation and results

## Our test-time adaptation method
ideas, implementation, thoughts, expected results and effective results



# Related Topics
Take a look if there's something useful in here.
### Domain Adaptation

 Likewise, one relevant
topic closely related to TTBA (batch size equals 1) is one-
shot domain adaptation (Luo et al., 2020; Varsavsky et al.,
2020  || .), which entails adapting to a single unlabeled instance
while still necessitating the source domain during adaptation.
Moreover, OTTA is closely related to online domain adap-
tation (Moon et al., 2020; Yang et al., 2022), which involves
adapting to an unlabeled target domain with streaming data
that is promptly deleted after adaptation.


Luo, Y., Liu, P., Guan, T., Yu, J., & Yang, Y. (2020). Adversarial style
mining for one-shot unsupervised domain adaptation. In Proceed-
ings of NeurIPS (pp. 20612–20623)

Varsavsky, T., Orbes-Arteaga, M., Sudre, C. H., Graham, M. S., Nachev,
P., & Cardoso, M. J. (2020). Test-time unsupervised domain adap-
tation. In Proceedings of MICCAI (pp. 428–436).

Moon, J. H., Das, D., Lee, C. S. G. (2020). Multi-step online unsu-
pervised domain adaptation. In Proceedings of ICASSP (pp.
41172–41576).

Yang, L., Gao, M., Chen, Z., Xu, R., Shrivastava, A., & Ramaiah, C.
(2022). Burn after reading: Online adaptation for cross-domain
streaming data. In Proceedings of ECCV (pp. 404–422).

### Domain Generalization
watch the paper, take a look of the quoted papers and see if there's something useful.

### 2.6 Test-Time Augmentation
Generally, test-time augmentation techniques do not explicitly consider distribution shifts but can be advantageous for TTA methods


# Bibliography


1. <a id="ref-liang2025"></a> Liang J., He R., Tan T. (2025). _A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts_. IJCV, 133, 31-64, [Link](https://doi.org/10.1007/s11263-024-02181-w)
2. <a id="ref-clip2021"></a> Radford A., Kim K. J., Hallacy C., et al. (2021). _Learning Transferable Visual Models From Natural Language Supervision_. ICML 2021, 8748-8763, [Link](https://arxiv.org/abs/2103.00020)
3. <a id="ref-tpt2022"></a> Shu M., Nie W., Huang D., et al. (2022). _Test-time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models_. NeurIPS 2022, [Link](https://arxiv.org/abs/2209.07511)


Ishii, M., & Sugiyama, M. (2021). Source-free domain adaptation via
distributional alignment by matching batch normalization statis-
tics. arXiv:2101.10842.

Iwasawa, Y., & Matsuo, Y. (2021). Test-time classifier adjustment mod-
ule for model-agnostic domain generalization. In Proceedings of
NeurIPS (pp. 2427–2440).

Jain, V., & Learned-Miller, E. (2011). Online domain adaptation of a
pre-trained cascade of classifiers. In Proceedings of CVPR (pp.
577–584).