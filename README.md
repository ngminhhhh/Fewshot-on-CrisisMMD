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

---

## Dataset

### Overview

We use the https://crisisnlp.qcri.org/crisismmd.html — a multimodal dataset consisting of images and tweets collected during natural disasters.

This work focuses on: **Task 2: Humanitarian Classification**

The goal is to classify each image-tweet pair into one of the following humanitarian categories:

- Infrastructure and utility damage  
- Not humanitarian  
- Rescue, volunteering, or donation effort  
- Other relevant information  
- Affected individuals  
- Injured or dead people  
- Vehicle damage  
- Missing or found people  

---

### Dataset Statistics

| Split | #Samples |
|------|---------|
| Train | `13608` |
| Val   | `2237` |
| Test  | `2237` |

---

### Class Distribution

#### Train Set
![Train Distribution](report_images/Fewshot/train_dis.png)

#### Validation Set
![Train Distribution](report_images/Fewshot/val_dis.png)

#### Test Set
![Train Distribution](report_images/Fewshot/test_dis.png)

---

### 🖼️ Sample Images (Train Set)

Each image below shows **5 randomly sampled examples per class**:

#### Infrastructure & Utility Damage
![infrastructure](report_images/Fewshot/infrastructure_and_utility_damage.png)

#### Not Humanitarian
![not_humanitarian](report_images/Fewshot/not_humanitarian.png)

#### Rescue / Volunteering / Donation Effort
![rescue](report_images/Fewshot/rescue_volunteering_or_donation_effort.png)

#### Other Relevant Information
![other](report_images/Fewshot/other_relevant_infomation.png)

#### Affected Individuals
![affected](report_images/Fewshot/affected_individuals.png)

#### Injured or Dead People
![injured](report_images/Fewshot/injured_or_dead_people.png)

#### Vehicle Damage
![vehicle](report_images/Fewshot/vehicle_damage.png)

#### Missing or Found People
![missing](report_images/Fewshot/missing_or_found_people.png)
---

## 2. Model

![Model Architecture](report_images/Fewshot/model.png)

### Model Description

We propose a multimodal few-shot classifier built on top of **CLIP (ViT-B/32)** as a frozen backbone.

---

#### Feature Extraction

- **Image Encoder**: Extract visual features from input images  
- **Text Encoder**: Encode class prompts into semantic embeddings  

Both encoders are frozen to leverage pretrained knowledge.

---

#### Multimodal Fusion

Image and text features are combined using a weighted sum:

f = α · f<sub>image</sub> + (1 - α) · f<sub>text</sub>
- α = **0.5**, giving equal importance to both modalities  
- Features are L2-normalized before fusion  

---

#### Projection Head

The fused representation is passed through an MLP projection head to learn task-specific features.

We experiment with two variants:

- **Model 1**:
  - Hidden layers: `[256, 256]`

- **Model 2**:
  - Hidden layers: `[256]`

Each layer uses:
- Linear transformation  
- ReLU activation  

---

#### Output Representation

- Final embeddings are L2-normalized  
- Used for classification via similarity or downstream classifier  

---

#### Key Design Choices

- Freeze CLIP → avoid overfitting in few-shot setting  
- Balanced fusion (α = 0.5) → combine visual + textual signals  
- Lightweight MLP → adapt features to task  

---

This architecture effectively leverages pretrained multimodal representations while remaining simple and efficient for few-shot learning.

### Prompt Engineering

We design prompts to align with humanitarian semantics:

```python
class_prompts = [
    "an image and tweet about infrastructure and utility damage after a disaster",
    "an image and tweet not related to humanitarian disaster",
    "an image and tweet about rescue, volunteering, or donation effort after a disaster",
    "an image and tweet about other relevant disaster information",
    "an image and tweet about affected individuals after a disaster",
    "an image and tweet about injured or dead people after a disaster",
    "an image and tweet about vehicle damage after a disaster",
    "an image and tweet about missing or found people after a disaster"
]
```

### Results
