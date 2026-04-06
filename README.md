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

Few-shot learning focuses on adapting models to new tasks with only a small number of labeled examples per class. This setting is especially useful when annotated data is limited.

---

## Dataset

### Overview

We use the https://crisisnlp.qcri.org/crisismmd.html dataset, a multimodal benchmark of disaster-related images and tweets.

This project focuses on **Task 2: Humanitarian Classification**, where each image-tweet pair is classified into one of the following categories:

- Infrastructure and utility damage  
- Not humanitarian  
- Rescue, volunteering, or donation effort  
- Other relevant information  
- Affected individuals  
- Injured or dead people  
- Vehicle damage  
- Missing or found people  

### Dataset Statistics

| Split | #Samples |
|------|---------:|
| Train | `13608` |
| Val   | `2237` |
| Test  | `2237` |

### Class Distribution

#### Train Set
<p align="center">
  <img src="report_images/Fewshot/train_dis.png" width="50%">
</p>

#### Validation Set
<p align="center">
  <img src="report_images/Fewshot/val_dis.png" width="50%">
</p>

#### Test Set
<p align="center">
  <img src="report_images/Fewshot/test_dis.png" width="50%">
</p>

### Sample Images (Train Set)

Each figure shows **5 sampled examples per class**.

#### Infrastructure & Utility Damage
<p align="center">
  <img src="report_images/Fewshot/infrastructure_and_utility_damage.png" width="60%">
</p>

#### Not Humanitarian
<p align="center">
  <img src="report_images/Fewshot/not_humanitarian.png" width="60%">
</p>

#### Rescue / Volunteering / Donation Effort
<p align="center">
  <img src="report_images/Fewshot/rescue_volunteering_or_donation_effort.png" width="60%">
</p>

#### Other Relevant Information
<p align="center">
  <img src="report_images/Fewshot/other_relevant_infomation.png" width="60%">
</p>

#### Affected Individuals
<p align="center">
  <img src="report_images/Fewshot/affected_individuals.png" width="60%">
</p>

#### Injured or Dead People
<p align="center">
  <img src="report_images/Fewshot/injured_or_dead_people.png" width="60%">
</p>

#### Vehicle Damage
<p align="center">
  <img src="report_images/Fewshot/vehicle_damage.png" width="60%">
</p>

#### Missing or Found People
<p align="center">
  <img src="report_images/Fewshot/missing_or_found_people.png" width="60%">
</p>

---

## Model

<p align="center">
  <img src="report_images/Fewshot/model.png" width="60%">
</p>

### Overview

We build a multimodal few-shot classifier using **CLIP (ViT-B/32)** as a frozen backbone.

### Feature Extraction

- **Image encoder** extracts visual features from input images  
- **Text encoder** encodes class prompts into semantic embeddings  

Both encoders are frozen to preserve pretrained knowledge and reduce overfitting.

### Multimodal Fusion

Image and text features are fused by weighted averaging:

`f = α · f_image + (1 - α) · f_text`

- α = **0.5**
- Both features are L2-normalized before fusion

### Projection Head

The fused representation is passed through a lightweight MLP with hidden dimensions:

`[256, 128]`

Each layer consists of:

- Linear
- ReLU

### Output Representation

- Final embeddings are L2-normalized  
- Used for classification through similarity or a downstream classifier  

### Design Choices

- **Frozen CLIP** to improve generalization in low-data settings  
- **Balanced fusion** to combine visual and textual information equally  
- **Lightweight MLP** to adapt pretrained features to the task  

### Prompt Engineering

We use class-specific prompts to align text embeddings with humanitarian semantics:

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

## Results
<p align="center">
  <img src="report_images/Fewshot/acc-loss.png" width="60%">
</p>
Epoch 3 has the highest accuracy on val set so we choice weights of epoch 3 is the best weights and use it to evaluate on test set 

| Dataset | Top-1 Accuracy | Top-5 Accuracy |
|--------|--------------|---------------|
| Train  | 70.10%       | 98.73%        |
| Val    | 62.49%       | 97.99%        |
| Test   | 61.51%       | 97.27%        |