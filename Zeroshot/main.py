import time
import torch
import clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import open_clip
from transformers import AutoModel, AutoProcessor, AlignModel, AlignProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

templates = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a black and white photo of a {}",
]

class_names = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute"
}


def sync_if_cuda():
    if device == "cuda":
        torch.cuda.synchronize()


def summarize_metrics(top1_correct, top5_correct, total, inference_time):
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    avg_time_per_image = inference_time / total
    throughput = total / inference_time if inference_time > 0 else 0.0

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "inference_time": inference_time,
        "avg_time_per_image": avg_time_per_image,
        "throughput": throughput,
    }


def clip_classifier(root, prompts):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    valset = ImageFolder(root=root, transform=preprocess)
    val_loader = DataLoader(
        valset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_classes = len(prompts) // len(templates)
    num_templates = len(templates)

    text_features = text_features.view(num_classes, num_templates, -1)
    text_features = text_features.mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    top1_correct = 0
    top5_correct = 0
    total = 0

    sync_if_cuda()
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T

            pred_top1 = logits.argmax(dim=1)
            top1_correct += (pred_top1 == labels).sum().item()

            k = min(5, logits.size(1))
            pred_top5 = logits.topk(k, dim=1).indices
            top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    sync_if_cuda()
    inference_time = time.perf_counter() - start_time

    return summarize_metrics(top1_correct, top5_correct, total, inference_time)


class SiglipTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)


def siglip_classifier(root, prompts):
    model_name = "google/siglip-base-patch16-224"

    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    valset = ImageFolder(root=root, transform=SiglipTransform(processor))
    val_loader = DataLoader(
        valset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    with torch.no_grad():
        text_inputs = processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        text_features = model.get_text_features(**text_inputs)

        if not isinstance(text_features, torch.Tensor):
            text_outputs = model.text_model(**text_inputs)
            text_features = text_outputs.pooler_output
            if hasattr(model, "text_projection") and model.text_projection is not None:
                text_features = model.text_projection(text_features)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_classes = len(prompts) // len(templates)
    num_templates = len(templates)

    text_features = text_features.view(num_classes, num_templates, -1)
    text_features = text_features.mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    top1_correct = 0
    top5_correct = 0
    total = 0

    sync_if_cuda()
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=(device == "cuda"))
            labels = labels.to(device, non_blocking=(device == "cuda"))

            image_features = model.get_image_features(pixel_values=images)

            if not isinstance(image_features, torch.Tensor):
                image_outputs = model.vision_model(pixel_values=images)
                image_features = image_outputs.pooler_output
                if hasattr(model, "visual_projection") and model.visual_projection is not None:
                    image_features = model.visual_projection(image_features)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T

            pred_top1 = logits.argmax(dim=1)
            top1_correct += (pred_top1 == labels).sum().item()

            k = min(5, logits.size(1))
            pred_top5 = logits.topk(k, dim=1).indices
            top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    sync_if_cuda()
    inference_time = time.perf_counter() - start_time

    return summarize_metrics(top1_correct, top5_correct, total, inference_time)


def openclip_classifier(root, prompts):
    model_name = "ViT-B-32"
    pretrained_name = "laion2b_s34b_b79k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_name
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    valset = ImageFolder(root=root, transform=preprocess)
    val_loader = DataLoader(
        valset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_classes = len(prompts) // len(templates)
    num_templates = len(templates)

    text_features = text_features.view(num_classes, num_templates, -1)
    text_features = text_features.mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    top1_correct = 0
    top5_correct = 0
    total = 0

    sync_if_cuda()
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T

            pred_top1 = logits.argmax(dim=1)
            top1_correct += (pred_top1 == labels).sum().item()

            k = min(5, logits.size(1))
            pred_top5 = logits.topk(k, dim=1).indices
            top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    sync_if_cuda()
    inference_time = time.perf_counter() - start_time

    return summarize_metrics(top1_correct, top5_correct, total, inference_time)


class AlignTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)


def align_classifier(root, prompts):
    model_name = "kakaobrain/align-base"

    model = AlignModel.from_pretrained(model_name).to(device)
    processor = AlignProcessor.from_pretrained(model_name)
    model.eval()

    valset = ImageFolder(root=root, transform=AlignTransform(processor))
    val_loader = DataLoader(
        valset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    with torch.no_grad():
        text_inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        text_outputs = model.get_text_features(**text_inputs)
        text_features = text_outputs.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_classes = len(prompts) // len(templates)
    num_templates = len(templates)

    text_features = text_features.view(num_classes, num_templates, -1)
    text_features = text_features.mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    top1_correct = 0
    top5_correct = 0
    total = 0

    sync_if_cuda()
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=(device == "cuda"))
            labels = labels.to(device, non_blocking=(device == "cuda"))

            image_outputs = model.get_image_features(pixel_values=images)
            image_features = image_outputs.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T

            pred_top1 = logits.argmax(dim=1)
            top1_correct += (pred_top1 == labels).sum().item()

            k = min(5, logits.size(1))
            pred_top5 = logits.topk(k, dim=1).indices
            top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    sync_if_cuda()
    inference_time = time.perf_counter() - start_time

    return summarize_metrics(top1_correct, top5_correct, total, inference_time)


if __name__ == "__main__":
    prompts = [
        template.format(name)
        for name in class_names.values()
        for template in templates
    ]

    root = "imagenette2-320/val"

    results = {
        "CLIP": clip_classifier(root, prompts),
        "SigLIP": siglip_classifier(root, prompts),
        "OpenCLIP": openclip_classifier(root, prompts),
        "ALIGN": align_classifier(root, prompts),
    }

    for model_name, scores in results.items():
        print(f"{model_name}:")
        print(f"  Top-1 Accuracy      : {scores['top1']:.4f}")
        print(f"  Top-5 Accuracy      : {scores['top5']:.4f}")
        print(f"  Total Infer Time    : {scores['inference_time']:.4f} s")
        print(f"  Avg Time / Image    : {scores['avg_time_per_image']*1000:.4f} ms")
        print(f"  Throughput          : {scores['throughput']:.2f} images/s")
        print()