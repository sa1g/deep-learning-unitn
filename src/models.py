import time
from typing import List
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from dataclasses import dataclass
from copy import deepcopy

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
        clip_model: clip.model.CLIP,
        base_prompt: str = "a photo of a [CLS]",
        device="cuda",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = class_names
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.token_embedding = clip_model.token_embedding
        self.token_embedding.requires_grad_(False)

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        # self.tokenizer = clip.tokenize

        self.__init_ctx_from_prompt(base_prompt=base_prompt)

    def __init_ctx_from_prompt(self, base_prompt: str) -> None:
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
        c_token_sot = torch.tensor([[49406]]).to(self.device)  # SOT
        c_token_eot = torch.tensor([[49407]]).to(self.device)  # EOT
        c_token_pad = torch.tensor([[0]]).to(self.device)  # PAD

        # Tokenize prefix, suffix and class names
        tokenized_prefix = self.tokenizer(promt_prefix).to(self.device)
        tokenized_suffix = self.tokenizer(promt_suffix).to(self.device)

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

        tokenized_class_names = self.tokenizer(self.class_names).to(self.device)

        # BASE full prompt
        # [CLS] + prefix + class_name + suffix + EOT
        # pre-computed as it's used for all classes and images :)
        self.tokenized_initial_full_prompt = self.tokenizer(
            [base_prompt.replace("[CLS]", c) for c in self.class_names]
        )

        # Get base embeddings
        with torch.no_grad():
            self.embedded_sot = self.token_embedding(c_token_sot)
            self.embedded_eot = self.token_embedding(c_token_eot)
            self.embedded_pad = self.token_embedding(c_token_pad)
            self.embedded_prefix = self.token_embedding(c_tokenized_prefix)
            self.embedded_suffix = self.token_embedding(c_tokenized_suffix)
            embedded_class_names = self.token_embedding(tokenized_class_names)
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
        self, class_names: List[str], arch: CLIPModels = CLIPModels.ViTB32, device="cuda"
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # clip_model: clip.model.CLIP
        clip_model, _ = clip.load(arch, device=self.device, jit=False)
        
        # clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
        # # model_name="ViT-B-16", pretrained="datacomp_xl_s13b_b90k", device=device#, force_quick_gelu=True
        # model_name="ViT-B-16", pretrained="openai", device=device, force_quick_gelu=True
        # )
        
        self.dtype = clip_model.visual.conv1.weight.dtype
        # self.clip = clip_model
        self.image_encoder = clip_model.visual
        self.image_encoder.eval() # added for safety, should be cool

        self.logit_scale = clip_model.logit_scale.data
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.transformer.eval() # added for safety, should be cool

        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        for _, param in self.named_parameters():
            param.requires_grad_(False)

        self.prompt_learner = TPTPromptLearner(
            class_names=class_names, clip_model=clip_model
        )

        self.class_names = class_names

        #

    def __encode_text(
        self, tokenized_prompt: torch.Tensor, embedded_prompt: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode the text prompt using the CLIP model.
            The tokenizer is external.

        Source: CLIP source code. model.py#L343
        """
        x = embedded_prompt + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2).contiguous()  # NLP -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # ORIGINAL:
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompt.argmax(dim=-1)]
            @ self.text_projection
        )

        # CUSTOM:
        # 120ms faster than the original
        # Suppose that all prompts are the same length and
        # EOT is the last token in the sequence
        # eot_pos = tokenized_prompt.size(1) -1
        # x = x[:, eot_pos, :] 
        # x = x @ self.text_projection
        
        return x

    def forward(self, image: torch.Tensor, is_image: bool = True) -> torch.Tensor:
        """
        Inference function for the CLIP model.

        Args:
            images (torch.Tensor): Input images.
            is_image (bool): whether the input is an iamge or already image_features.
                If False, the input is assumed to be already image features.
        Returns:
            logits (torch.Tensor): Logits from the CLIP model.
        """
        if is_image:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            image_features = image

        # embedded_prompt = self.prompt_learner().type(self.dtype)

        prompt = "a photo of a {}"

        embedded_prompt = torch.cat(
            [
                self.prompt_learner.tokenizer(prompt.format(c))
                for c in self.class_names
            ]
        ).to(self.device)

        embedded_prompt = self.prompt_learner.token_embedding(
            embedded_prompt).type(self.dtype)

        txt_features = self.__encode_text(
            self.prompt_learner.tokenized_initial_full_prompt, embedded_prompt
        )
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ txt_features.t()

        if is_image:
            return logits, image_features
        else:
            return logits

    def reset(self):
        """
        Reset the prompt learner to its initial state.
        """
        self.prompt_learner.reset()


class TPT(nn.Module):
    def __init__(
        self,
        class_names: List[str],
        tta_steps: int = 1,
        lr: float = 0.0001,
        arch: CLIPModels = CLIPModels.ViTB32,
        device="cuda",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tpt_steps = tta_steps

        model = TPTModel(
            class_names=class_names,
            # arch=arch,
            arch="ViT-B/16",
            device=self.device,
        )
        self.model = model.to(self.device)

        # Get all trainable parameters (filter by requires_grad)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Initialize optimizer with trainable parameters
        self.optim = torch.optim.AdamW(trainable_params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()

        self.optim_init = deepcopy(self.optim.state_dict())

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
        for _ in range(self.tpt_steps):
            with torch.cuda.amp.autocast():
                logits, image_features = self.model(input)

                # Select the most confident samples
                if selected_idx is not None:
                    logits = logits[selected_idx]
                else:
                    logits, selected_idx = self.__select_confident_samples(logits)

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

    def __reset(self) -> None:
        """Full reset of prompt learner and optimizer state"""
        # 1. Reset prompt embeddings
        self.model.reset()

        # # 2. Reset optimizer state
        self.optim.load_state_dict(deepcopy(self.optim_init))

        # # 3. Reset gradient scaler if using AMP
        # if hasattr(self, "scaler"):
        #     self.scaler.load_state_dict(torch.cuda.amp.GradScaler().state_dict())
