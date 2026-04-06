from dataloader.Dataloader import *
from model.Classifier import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model, dataloader, class_features, loader_name="Evaluation"):
    model.eval()

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=loader_name)

        for batch in pbar:
            images = batch["image"]
            texts = batch["text"]
            labels = batch["label"].to(model.device)

            # Forward
            sample_features = model(images, texts)   # [B, D]
            logits = sample_features @ class_features.T   # [B, num_classes]

            # Loss
            batch_size = labels.size(0)
            total += batch_size

            # Top-1
            pred_top1 = logits.argmax(dim=1)
            top1_correct += (pred_top1 == labels).sum().item()

            # Top-5
            k = min(5, logits.size(1))  # tránh lỗi nếu số lớp < 5
            pred_top5 = logits.topk(k, dim=1).indices   # [B, k]
            top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            # Progress bar
            pbar.set_postfix({
                "top1_acc": top1_correct / total,
                "top5_acc": top5_correct / total
            })

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    return top1_acc, top5_acc

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier(out_dim=512).to(device) 
    model.load_state_dict(torch.load("./weights/weights.pth"))

    train_tsv_file = "data/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv"
    val_tsv_file = "data/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv"
    test_tsv_file = "data/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv"
    img_dir = "data/CrisisMMD_v2.0"

    train_dataset = CrisisDataset(train_tsv_file, "data/CrisisMMD_v2.0", model.clip_preprocess)
    val_dataset = CrisisDataset(val_tsv_file, "data/CrisisMMD_v2.0", model.clip_preprocess)
    test_dataset = CrisisDataset(test_tsv_file, "data/CrisisMMD_v2.0", model.clip_preprocess)

    class_names = [
        "infrastructure and utility damage",
        "not humanitarian",
        "rescue volunteering or donation effort",
        "other relevant information",
        "affected individuals",
        "injured or dead people",
        "vehicle damage",
        "missing or found people"
    ]

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

    with torch.no_grad():
        class_tokens = clip.tokenize(class_prompts, truncate=True).to(model.device)
        class_features = model.clip_encoder.encode_text(class_tokens)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        class_features = class_features.float()

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    eval_lst = [("train", train_loader), ("val", val_loader), ("test", test_loader)]

    for (name, dataloader) in eval_lst:
        top1_acc, top5_acc = evaluate(model, dataloader, class_features, name)
        print(f"{name}: top1_acc={top1_acc}, top5_acc={top5_acc}")

 