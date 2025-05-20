from typing import List
import numpy as np
import open_clip
import open_clip.transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from copy import deepcopy

from open_clip.transformer import text_global_pool


@dataclass(frozen=True)
class CLIPModels:
    ViTB32: str = "ViT-B/32"
    # You can add more, but the `kornia_preprocess` should be modified accordingly
    # ViTB16: str = "ViT-B/16"
    # RN50: str = "RN50"


class TPTPromptLearner(nn.Module):
    def __init__(
        self,
        class_names: List[str],
        clip_model: open_clip.model.CLIP,
        arch: CLIPModels = CLIPModels.ViTB32,
        base_prompt: str = "a photo of a [CLS]",
        device="cuda",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = class_names

        tokenizer = open_clip.get_tokenizer(arch)

        self.__init_ctx_from_prompt(
            tokenizer=tokenizer,
            token_embedding=clip_model.token_embedding,
            base_prompt=base_prompt,
        )

    def __init_ctx_from_prompt(
        self, tokenizer, token_embedding, base_prompt: str
    ) -> None:
        """
        Initialize the context tokens from the base prompt.

        We need to make sure that the CLS token is NOT "exploded" in the prompt.

        The idea is to have prompts tuned without having to manually manage where the CLS token is.

        To do this we need to keep the CLS token position in the prompt, and update it accordingly
        when needed.

        I'm splitting the prompt into prefix and suffix, using [CLS] as a separator.
        They are trained as two different parameters, and then concatenated together.

        """

        # Split the base prompt into prefix and suffix
        promt_prefix = base_prompt.split("[CLS]")[0]
        promt_suffix = base_prompt.split("[CLS]")[1]

        # "Clean" PAD, SOT and EOT tokens
        c_token_sot = torch.tensor([[tokenizer.sot_token_id]]).to(self.device)
        c_token_eot = torch.tensor([[tokenizer.eot_token_id]]).to(self.device)
        c_token_pad = torch.tensor([[0]]).to(self.device)  # PAD

        # Tokenize prefix, suffix and class names
        tokenized_prefix = tokenizer(promt_prefix).to(self.device)
        tokenized_suffix = tokenizer(promt_suffix).to(self.device)

        # remove PAD, SOT and EOT tokens
        # Extract "clean" tokens
        c_tokenized_prefix = tokenized_prefix[
            (tokenized_prefix != c_token_sot)
            & (tokenized_prefix != c_token_eot)
            & (tokenized_prefix != c_token_pad)
        ].to(self.device)
        c_tokenized_suffix = tokenized_suffix[
            (tokenized_suffix != c_token_sot)
            & (tokenized_suffix != c_token_eot)
            & (tokenized_suffix != c_token_pad)
        ].to(self.device)

        tokenized_class_names = tokenizer(self.class_names).to(self.device)

        # BASE full prompt
        # [CLS] + prefix + class_name + suffix + EOT
        # pre-computed as it's used for all classes and images :)
        self.tokenized_initial_full_prompt = tokenizer(
            [base_prompt.replace("[CLS]", c) for c in self.class_names]
        )

        # Get base embeddings
        with torch.no_grad():
            self.embedded_sot = token_embedding(c_token_sot)
            self.embedded_eot = token_embedding(c_token_eot)
            self.embedded_pad = token_embedding(c_token_pad)
            self.embedded_prefix = token_embedding(c_tokenized_prefix)
            self.embedded_suffix = token_embedding(c_tokenized_suffix)
            embedded_class_names = token_embedding(tokenized_class_names)
            self.embedded_max_len = embedded_class_names.shape[1]

        # Setup clean embedded_class_names (list)
        # Mask to filter out SOT/EOT/PAD tokens (shape [200, 77])
        mask = (
            (tokenized_class_names != c_token_sot)
            & (tokenized_class_names != c_token_eot)
            & (tokenized_class_names != c_token_pad)
        )

        # Apply mask to embeddings (for each class)
        clean_embeddings = []
        for i in range(embedded_class_names.shape[0]):
            # masked_select would flatten, so we use boolean indexing
            clean_embed = embedded_class_names[i][mask[i]]  # [num_valid_tokens, 512]
            clean_embeddings.append(
                clean_embed.unsqueeze(0)
            )  # [1, num_valid_tokens, 512]

        self.embedded_class_names = clean_embeddings

        for i, embed in enumerate(clean_embeddings):
            self.register_buffer(f"class_embed_{i}", embed)

        # Create "init" states and set learnable parameters
        self.init_state_prefix = self.embedded_prefix.detach().clone()
        self.init_state_suffix = self.embedded_suffix.detach().clone()
        self.embedded_prefix = nn.Parameter(self.embedded_prefix)
        self.embedded_suffix = nn.Parameter(self.embedded_suffix)
        self.register_parameter("embedded_prefix", self.embedded_prefix)
        self.register_parameter("embedded_suffix", self.embedded_suffix)

    def forward(self) -> torch.Tensor:
        prompts = []
        for i in range(len(self.class_names)):

            # embeddeD_max_len: 77
            # embedded_prefix: torch.Size([4, 512])
            # embedded_class_names: torch.Size([1, 1, 512])
            # embedded_suffix: torch.Size([0, 512]

            padding_size = (
                self.embedded_max_len
                - self.embedded_prefix.shape[0]
                - getattr(self, f"class_embed_{i}").shape[1]
                - self.embedded_suffix.shape[0]
            ) - 2  # # -2 for SOT and EOT

            ## embedded sot shape: torch.Size([1, 1, 512])
            ## embedded prefix shape: torch.Size([1, 4, 512])
            ## embedded class names shape: torch.Size([1, 1, 1, 512])
            ## embedded suffix shape: torch.Size([1, 0, 512])
            ## embedded eot shape: torch.Size([1, 1, 512])
            ## effective padding shape: torch.Size([1, 70, 512])
            ## Prompt shape: torch.Size([1, 77, 512])

            prompt = torch.cat(
                (
                    self.embedded_sot,
                    self.embedded_prefix.unsqueeze(0),
                    # self.embedded_class_names[i],
                    getattr(self, f"class_embed_{i}"),
                    self.embedded_suffix.unsqueeze(0),
                    self.embedded_eot,
                    self.embedded_pad.repeat(1, padding_size, 1),
                ),
                dim=1,
            )

            prompts.append(prompt)

        prompts = torch.cat(prompts, dim=0)
        # Must have shape torch.Size([200, 77, 512]) (classes, feature1, feature2)
        return prompts

    def reset(self) -> None:
        # TODO: check, doin without `data`

        # self.embedded_prefix.data.copy_(self.init_state_prefix)
        # self.embedded_suffix.data.copy_(self.init_state_suffix)
        with torch.no_grad():
            self.embedded_prefix.copy_(self.init_state_prefix)
            self.embedded_suffix.copy_(self.init_state_suffix)


class TPTModel(nn.Module):
    def __init__(
        self,
        class_names: List[str],
        arch: CLIPModels,
        pretrained: str,
        device="cuda",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        clip_model: open_clip.model.CLIP
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=arch,
            pretrained=pretrained,
            device=device,
            force_quick_gelu=True,
        )

        self.model = clip_model
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(arch)
        self.class_names = class_names

        self.visual : open_clip.transformer.VisionTransformer = clip_model.visual
        self.visual.eval()

        self.token_embedding = clip_model.token_embedding

        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask
        self.text_pool_type = clip_model.text_pool_type

        for _, param in self.named_parameters():
            param.requires_grad_(False)

        self.prompt_learner = TPTPromptLearner(
            arch=arch, class_names=class_names, clip_model=clip_model
        )

        # # 512
        # text_transformer: open_clip.transformer.Transformer = self.transformer
        # text_out_ln: open_clip.transformer.LayerNorm = self.ln_final

        # for block in text_transformer.resblocks:
        #     print(block.ln_1.weight.shape)
        #     print(block.ln_2.weight.shape)
        # print(text_out_ln.weight.shape)

        # # 768
        # image_transformer: open_clip.transformer.Transformer = self.visual.transformer
        # image_out_ln: open_clip.transformer.LayerNorm = self.visual.ln_post

        # for block in image_transformer.resblocks:
        #     print(block.ln_1.weight.shape)
        #     print(block.ln_2.weight.shape)
        # print(image_out_ln.weight.shape)

        # exit()

    def __encode_image(self, image, normalize: bool = False) -> torch.Tensor:
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def __encode_text(self, text=None, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.prompt_learner().to(cast_dtype)

        text = self.prompt_learner.tokenized_initial_full_prompt

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inference function for the CLIP model.

        Args:
            images (torch.Tensor): Input images.
        Returns:
            logits (torch.Tensor): Logits from the CLIP model.
        """

        with torch.no_grad():
            image_features = self.__encode_image(image, normalize=True)

        text_features = self.__encode_text(normalize=True)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features

    def reset(self):
        """
        Reset the prompt learner to its initial state.
        """
        self.prompt_learner.reset()


class TPT(nn.Module):
    def __init__(
        self,
        pretrained: str,
        arch: CLIPModels,
        class_names: List[str],
        tta_steps: int = 1,
        lr: float = 0.0001,
        device="cuda",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tpt_steps = tta_steps

        model = TPTModel(
            class_names=class_names,
            arch=arch,
            pretrained=pretrained,
            device=self.device,
        )
        self.model = model.to(self.device)
        self.model.eval()

        # TEST
        self.model.visual.ln_post.requires_grad_(True)
        self.model.ln_final.requires_grad_(True)

        # Get all trainable parameters (filter by requires_grad)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Initialize optimizer with trainable parameters
        self.optim = torch.optim.AdamW(trainable_params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()

        self.optim_init = deepcopy(self.optim.state_dict())

        # Initialize backup lists
        self.ln_backup = {
            'weights': [],  # For gamma (scale)
            'biases': []    # For beta (shift)
        }

        # Backup all LN params in text encoder
        for block in self.model.transformer.resblocks:
            self.ln_backup['weights'].append(block.ln_1.weight.data.detach().clone())  # gamma for ln_1
            self.ln_backup['biases'].append(block.ln_1.bias.data.detach().clone())     # beta for ln_1
            self.ln_backup['weights'].append(block.ln_2.weight.data.detach().clone())  # gamma for ln_2
            self.ln_backup['biases'].append(block.ln_2.bias.data.detach().clone())      # beta for ln_2

        # Backup final LN
        self.ln_backup['weights'].append(self.model.ln_final.weight.data.detach().clone())
        self.ln_backup['biases'].append(self.model.ln_final.bias.data.detach().clone())

    def set_tta_steps(self, tta_steps: int) -> None:
        """
        Set the number of TTA steps.

        Args:
            tta_steps (int): Number of TTA steps.
        """
        self.tpt_steps = tta_steps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # manage Prompt learner finetuning
        # so do n iterations with fine tuning
        # then return the single prediction
        # loss, etc, are 100% internal

        selected_idx = None

        for step in range(self.tpt_steps):
            with torch.cuda.amp.autocast():
                logits, image_features = self.model(input)

                # Select the most confident samples
                if selected_idx is not None:
                    logits = logits[selected_idx]
                    
                else:
                    logits, selected_idx = self.__select_confident_samples(logits)
                    # mu, sigma = self.compute_stats(image_features[selected_idx])

                ## DAVIDE QUI
                # if step == 0:
                # # Adapt the layer norm parameters
                #     # for block in self.model.transformer.resblocks:                        
                #     #     self.adapt_ln_params(block.ln_1, mu, sigma, mode="hybrid")
                #     #     self.adapt_ln_params(block.ln_2, mu, sigma, mode="hybrid")
                #     self.adapt_ln_params(self.model.ln_final, mu, sigma, mode="scale") 
                

                # Compute the average entropy loss
                loss = self.__avg_entropy_loss(logits)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()


        # Actual inference
        with torch.no_grad(), torch.autocast("cuda"):
            # take only the last image of the input
            input = input[-1].unsqueeze(0)
            logits, _ = self.model(input)

            marginal_prob = F.softmax(logits, dim=1).mean(0)
            pred_class = marginal_prob.argmax().item()

        self.__reset()

        return pred_class
        # return logits

    def __select_confident_samples(
        self, logits: torch.Tensor, top: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Selects the top-k samples with the lowest entropy from the logits.

        Args:
            logits (torch.Tensor): The logits from the model.
            top (float): The fraction of samples to select.
                For example, if top=0.1, it selects the top 10% of samples.
        Returns:
            torch.Tensor: The selected logits.
            torch.Tensor: The indices of the selected samples.

        [Reference](https://github.com/azshue/TPT/blob/63ecbace79694205d7884e63fdc3137a200f0b0e/tpt_classification.py#L41C5-L41C11)
        """
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[
            : int(batch_entropy.size()[0] * top)
        ]

        return logits[idx], idx

    def __avg_entropy_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the average entropy of the model's outputs.
        Args:
            outputs (torch.Tensor): The model's outputs.
        Returns:
            torch.Tensor: The average entropy.

        [Reference](https://github.com/azshue/TPT/blob/63ecbace79694205d7884e63fdc3137a200f0b0e/tpt_classification.py#L46)
        """
        logits = outputs - outputs.logsumexp(
            dim=-1, keepdim=True
        )  # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(
            logits.shape[0]
        )  # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)

        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

    def compute_stats(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean (μ) and variance (σ²) for features.
        """
        mu = features.mean(dim=0, keepdim=True)
        var = features.var(dim=0, keepdim=True, unbiased=False)  # Match LN's behavior
        
        sigma = torch.sqrt(var + 1e-6)  # Avoid division by zero

        return mu.squeeze(0), sigma.squeeze(0) # [512], [512]

    def adapt_ln_params(
            self,
            ln_layer: torch.nn.LayerNorm,
            mu_new: torch.Tensor,
            sigma_new: torch.Tensor,
            mode: str = "scale",
        ):
            """
            Args:
                ln_layer: LayerNorm module to adapt.
                mu_new: Computed mean (1, 1) or (B, 1).
                sigma_new: Computed std (1, 1) or (B, 1).
                mode: How to adapt:
                    - "scale": γ* = γ * σ_new, β* = β + μ_new (lightweight)
                    - "replace": γ* = 1, β* = 0 (override)
                    - "hybrid": Normalize by (x-μ)/σ, then apply original γ/β
            """
            if mode == "scale":
                ln_layer.weight.data *= sigma_new.squeeze()  # γ* = γ * σ
                ln_layer.bias.data += mu_new.squeeze()  # β* = β + μ
            elif mode == "replace":
                ln_layer.weight.data.fill_(1.0)  # γ* = 1
                ln_layer.bias.data.fill_(0.0)  # β* = 0
            elif mode == "hybrid":
                # Store stats for custom forward (see below)
                ln_layer.mu_new = mu_new
                ln_layer.sigma_new = sigma_new
                # Override forward (optional)
                original_forward = ln_layer.forward

                def custom_forward(x):
                    x_norm = (x - ln_layer.mu_new) / (ln_layer.sigma_new + 1e-6)
                    return x_norm * ln_layer.weight + ln_layer.bias

                ln_layer.forward = custom_forward

    def __reset(self) -> None:
        """Full reset of prompt learner and optimizer state"""
        # 1. Reset prompt embeddings
        for p in self.model.parameters():
            p.grad = None
        
        self.model.reset()

        # with torch.no_grad():
        #     self.embedded_prefix.copy_(self.init_state_prefix)
        #     self.embedded_suffix.copy_(self.init_state_suffix)

        with torch.no_grad():
        #     idx = 0
        #     # Reset LN params in text encoder
        #     for block in self.model.transformer.resblocks:
        #         block.ln_1.weight.data.copy_(self.ln_backup['weights'][idx].clone())
        #         block.ln_1.bias.data.copy_(self.ln_backup['biases'][idx].clone())
        #         idx += 1
        #         block.ln_2.weight.data.copy_(self.ln_backup['weights'][idx].clone())
        #         block.ln_2.bias.data.copy_(self.ln_backup['biases'][idx].clone())
        #         idx += 1
        
            # # Reset final LN
            self.model.ln_final.weight.data.copy_(self.ln_backup['weights'][-1].clone())
            self.model.ln_final.bias.data.copy_(self.ln_backup['biases'][-1].clone())


        # # 2. Reset optimizer state
        self.optim.load_state_dict(deepcopy(self.optim_init))
