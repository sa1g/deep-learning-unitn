import open_clip
import torch

if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = open_clip.load("ViT-B/32", device=device, jit=False)
    model, _ ,_ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.token_embedding
    print("Model loaded successfully.")