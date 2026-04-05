import torch
import torch.nn as nn
import clip

class Classifier(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_encoder, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_encoder.eval() # Dont train clip encoder

        head_structure = []
        hidden_dims = [256, 128, 128]
        in_dim = self.clip_encoder.visual.output_dim

        for h in hidden_dims:
            head_structure.append(nn.Linear(in_dim, h))
            head_structure.append(nn.ReLU())
            in_dim = h

        head_structure.append(nn.Linear(hidden_dims[-1], out_dim))
        self.head = nn.Sequential(*head_structure)
        self.alpha = 0.5

    def forward(self, image, text):
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)   # text: list[str]
            text_features = self.clip_encoder.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image = image.to(self.device)  # image: tensor [B, C, H, W]
            image_features = self.clip_encoder.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        fused_features = self.alpha * image_features + (1 - self.alpha) * text_features
        fused_features = fused_features.float()
        logits = self.head(fused_features)

        return logits