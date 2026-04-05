# Zero-shot & Few-shot Learning 

## Zero-shot Learning

### Problem Overview
Zero-shot learning is a machine learning paradigm in which a model is required to recognize or classify data from classes that were **not seen during training**. Instead of relying on labeled examples for every class, the model leverages **semantic knowledge**, such as natural language descriptions, text embeddings, or attribute-based representations, to infer relationships between input data and unseen classes.

This approach is particularly useful in scenarios where:
- The number of classes is very large
- New classes frequently appear
- Labeling data is expensive or impractical

Typical applications include:
- Image classification
- Text classification
- Multimodal learning

In this project, we evaluate zero-shot performance using the **Imagenette dataset** with the following models:
- CLIP
- SigLIP
- OpenCLIP
- ALIGN

---

## Imagenette Dataset

### Introduction
Imagenette is a **lightweight image classification dataset** derived from ImageNet, introduced by fast.ai for rapid experimentation. It contains **10 selected classes**, including:

- Tench  
- English springer  
- Cassette player  
- Chain saw  
- Church  
- French horn  
- Garbage truck  
- Gas pump  
- Golf ball  
- Parachute  

Compared to the full ImageNet dataset, Imagenette is significantly smaller, making it suitable for:
- Quick prototyping
- Model comparison
- Educational purposes
- Experiments with limited computational resources

---

### Exploratory Data Analysis (EDA)

#### Number of samples

| Split | Number of samples |
|------|------------------|
| Train | 9,469 |
| Validation | 3,925 |

---

#### Class Distribution in Val Set

![Class Distribution](report_images/Zeroshot/class_dist.png)

---

#### Width&Height/Ratio statistic
![Width & Height Distribution](report_images/Zeroshot/width_height_dis.png)
![Ratio Distribution](report_images/Zeroshot/ratio.png)

#### Sample images by class
<img src="report_images/Zeroshot/tench.png" width="1400"/>
<img src="report_images/Zeroshot/English_springer.png" width="1400"/>
<img src="report_images/Zeroshot/Cassette_player.png" width="1400"/>
<img src="report_images/Zeroshot/chainsaw.png" width="1400"/>
<img src="report_images/Zeroshot/church.png" width="1400"/>
<img src="report_images/Zeroshot/French_horn.png" width="1400"/>
<img src="report_images/Zeroshot/garbage_truck.png" width="1400"/>
<img src="report_images/Zeroshot/gaspump.png" width="1400"/>
<img src="report_images/Zeroshot/golf_ball.png" width="1400"/>
<img src="report_images/Zeroshot/parachute.png" width="1400"/>


## Experiment results
### Results

| Model     | Params | Top-1 Acc | Top-5 Acc | Total Time (s) | Avg Time / Img (ms) | Throughput (img/s) |
|-----------|--------|----------|----------|----------------|---------------------|--------------------|
| CLIP      | ~151M  | 0.9896   | 0.9990   | **23.85**      | **6.08**            | **164.54**         |
| SigLIP    | ~428M  | 0.9944   | 0.9997   | 131.41         | 33.48               | 29.87              |
| OpenCLIP  | ~151M  | 0.9911   | 0.9997   | 31.63          | 8.06                | 124.11             |
| ALIGN     | ~1.8B  | **0.9975** | **1.0000** | 103.63         | 26.40               | 37.88              |

### Key Observations

- **ALIGN** achieves the highest accuracy, likely due to its significantly larger scale (~1.8B parameters).

- **CLIP** provides the best efficiency:
  - Fastest inference
  - Highest throughput

- **SigLIP** improves accuracy over CLIP but is much slower.

- **OpenCLIP** offers a good trade-off between speed and performance.

## 2. Few-shot Learning

Few-shot learning aims to train models that can generalize to new tasks using **only a small number of labeled examples per class**. This setting is more practical than traditional supervised learning in many real-world applications where labeled data is scarce.

In contrast to zero-shot:
- Zero-shot → no examples
- Few-shot → very few examples (e.g., 1–10 samples per class)

Few-shot learning typically relies on:
- Transfer learning
- Pretrained foundation models
- Metric learning or prompt-based approaches

---


## Summary

- Zero-shot learning enables classification without labeled samples for target classes.
- Imagenette provides a lightweight benchmark for experimentation.
- The dataset is balanced and suitable for evaluation.
- Few-shot learning complements zero-shot by allowing minimal supervision.

---