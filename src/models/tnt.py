import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class TNT(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        class_labels: dict,
        prompt: str = "a photo of a {}",
        device: str = "cuda",
        tnt_steps: int = 3,
        top_k: float = 0.1,
        epsilon: float = 1 / 255,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 7e-3,
    ):
        super().__init__()
        self.device = device
        self.model = model
        self.logit_scale = model.logit_scale.data.exp()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.tnt_steps = tnt_steps
        self.top_k = top_k
        self.eps = epsilon
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        with torch.no_grad():
            prompts = torch.cat(
                [self.tokenizer(prompt.format(c)) for c in class_labels.values()]
            ).to(device)
            self.text_features = model.encode_text(prompts, normalize=True)

        self.noise = None

    def reset(self):
        self.noise = None

    def select_confident_samples(self, logits: torch.Tensor, top: float = 0.1):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[
            : int(batch_entropy.size()[0] * top)
        ]
        return logits[idx], idx

    def entropy_minimization_loss(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0)
        return -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

    def inter_view_consistency_loss(self, features: torch.Tensor):
        pairwise_dist = torch.cdist(features, features, p=2)
        K = features.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=features.device)
        return pairwise_dist[mask].mean()  # Use mean instead of sum for stability

    def forward(self, x: torch.Tensor) -> int:
        x = x.to(self.device)

        # Initialize noise for all spatial dimensions
        if self.noise is None:
            self.noise = torch.randn_like(x[0], requires_grad=True, device=self.device)
            self.noise.data = self.noise.clamp(-self.eps, self.eps)

        with torch.autocast(self.device):
            for _ in range(self.tnt_steps):
                self.noise.requires_grad = True

                # Apply same noise to all views (alternative: could generate separate noise per view)
                x_aug = x + self.noise.unsqueeze(0)  # Add noise to all views
                x_aug = torch.clamp(x_aug, 0, 1)  # Clamp pixel values

                image_features = self.model.encode_image(x_aug, normalize=True)
                logits = self.logit_scale * image_features @ self.text_features.t()

                top_logits, top_idx = self.select_confident_samples(
                    logits, top=self.top_k
                )
                top_features = image_features[top_idx]

                entropy_loss = self.entropy_minimization_loss(top_logits)
                inter_view_loss = self.inter_view_consistency_loss(
                    top_features
                )  # Use features, not logits

                loss = self.alpha * entropy_loss + self.beta * inter_view_loss
                loss.backward()

                with torch.no_grad():
                    grad = self.noise.grad
                    self.noise -= self.lr * grad.sign()
                    self.noise.clamp_(-self.eps, self.eps)
                    self.noise.grad = None

        # Final inference
        with torch.no_grad():
            x_aug = x + self.noise.clamp(-self.eps, self.eps)
            image_features = self.model.encode_image(x_aug, normalize=True)
            logits = self.logit_scale * image_features @ self.text_features.t()

            # Select top-K views and apply temperature scaling before averaging
            top_logits, top_idx = self.select_confident_samples(logits, top=self.top_k)
            probs = F.softmax(top_logits / self.temperature, dim=1).mean(dim=0)

            pred_class = int(probs.argmax().item())
            self.reset()

        return pred_class


class TNTTop10(TNT):
    def __init__(
        self,
        model: nn.Module,
        class_labels: dict,
        prompt: str = "a photo of a {}",
        device: str = "cuda",
        tnt_steps: int = 3,
        top_k: float = 0.1,
        epsilon: float = 1 / 255,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 7e-3,
    ):
        super().__init__(
            model,
            class_labels,
            prompt,
            device,
            tnt_steps,
            top_k,
            epsilon,
            lr,
            alpha,
            beta,
            temperature,
        )

    def forward(self, x: torch.Tensor) -> int:
        x = x.to(self.device)

        # Initialize noise for all spatial dimensions
        if self.noise is None:
            self.noise = torch.randn_like(x[0], requires_grad=True, device=self.device)
            self.noise.data = self.noise.clamp(-self.eps, self.eps)

        with torch.autocast(self.device):
            for _ in range(self.tnt_steps):
                self.noise.requires_grad = True

                # Apply same noise to all views (alternative: could generate separate noise per view)
                x_aug = x + self.noise.unsqueeze(0)  # Add noise to all views
                x_aug = torch.clamp(x_aug, 0, 1)  # Clamp pixel values

                image_features = self.model.encode_image(x_aug, normalize=True)
                logits = self.logit_scale * image_features @ self.text_features.t()

                top_logits, top_idx = self.select_confident_samples(
                    logits, top=self.top_k
                )
                top_features = image_features[top_idx]

                entropy_loss = self.entropy_minimization_loss(top_logits)
                inter_view_loss = self.inter_view_consistency_loss(
                    top_features
                )  # Use features, not logits

                loss = self.alpha * entropy_loss + self.beta * inter_view_loss
                loss.backward()

                with torch.no_grad():
                    grad = self.noise.grad
                    self.noise -= self.lr * grad.sign()
                    self.noise.clamp_(-self.eps, self.eps)
                    self.noise.grad = None

        # Final inference
        with torch.no_grad():
            x_aug = x + self.noise.clamp(-self.eps, self.eps)
            image_features = self.model.encode_image(x_aug, normalize=True)
            logits = self.logit_scale * image_features @ self.text_features.t()

            # Select top-K views and apply temperature scaling before averaging
            top_logits, top_idx = self.select_confident_samples(
                logits[:-1:], top=self.top_k
            )
            top_logits = torch.cat((top_logits, logits[-1:]), dim=0)
            probs = F.softmax(top_logits / self.temperature, dim=1).mean(dim=0)

            pred_class = int(probs.argmax().item())
            self.reset()

        return pred_class
