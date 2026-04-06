import torch
import torch.nn as nn
import clip

import torch
import torch.nn as nn
import clip

class Classifier(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_encoder, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_encoder.eval()

        for p in self.clip_encoder.parameters():
            p.requires_grad = False

        hidden_dims = [256, 128]
        in_dim = self.clip_encoder.visual.output_dim

        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, out_dim))
        self.proj = nn.Sequential(*layers)

        self.alpha = 0.5

    def forward(self, image, text):
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            text_features = self.clip_encoder.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.float()

            image = image.to(self.device)
            image_features = self.clip_encoder.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.float()

        fused_features = self.alpha * image_features + (1 - self.alpha) * text_features
        fused_features = self.proj(fused_features)
        fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)

        return fused_features