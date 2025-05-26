from matplotlib import pyplot as plt
import torch

torch.manual_seed(456)
torch.cuda.manual_seed(456)

import torch.nn as nn
import torch.nn.functional as F
import open_clip

from src.utils import bench


from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess, kornia_random_crop
from src.data import ResnetA

from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF



class ClipWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        class_labels: dict,
        prompt: str = "a photo of a {}",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.class_labels = class_labels

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model = model
        self.logit_scale = model.logit_scale.exp()

        # Precompute text features
        with torch.no_grad():
            prompts = torch.cat(
                [self.tokenizer(prompt.format(c)) for c in class_labels.values()]
            ).to(device)
            self.text_features = model.encode_text(prompts, normalize=True)

        self.kmeans = KMeans(n_clusters=4, random_state=42)

    def forward(self, x: torch.Tensor) -> int:
        with torch.no_grad(), torch.autocast("cuda"):
            # x: (B, 3, 224, 224)
            image_features = self.model.encode_image(x, normalize=True)
        
            # Move to CPU and convert to numpy for sklearn
            features_np = image_features.cpu().numpy()
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_np)

            # Cluster features
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=6, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            ###################################################

            # # get the cluster with higher confidence
            # cluster_confidences = []
            # for cluster_idx in range(6):
            #     cluster_features = image_features[cluster_labels == cluster_idx]
            #     logits = self.logit_scale * cluster_features @ self.text_features.t()
            #     cluster_confidences.append(logits.mean().item())

            # # Get the cluster with the highest confidence
            # best_cluster_idx = cluster_confidences.index(max(cluster_confidences))

            # image_features_r = image_features[cluster_labels == best_cluster_idx]

            # image_features_r = torch.cat((image_features_r, image_features[-1:]), dim=0)

            # logits = self.logit_scale * image_features_r @ self.text_features.t()

            # marginal_prob = F.softmax(logits, dim=1).mean(0)

            # pred_class = marginal_prob.argmax().item()


            ###################################################

            # Cluster closer to the original image
            cluster_confidences = []
            for cluster_idx in range(6):
                cosine_sim = image_features[cluster_labels == cluster_idx] @ image_features[-1:].t()
                cluster_confidences.append(cosine_sim.mean().item())

            # Get the cluster with the highest confidence
            best_cluster_idx = cluster_confidences.index(max(cluster_confidences))

            image_features_r = image_features[cluster_labels == best_cluster_idx]

            image_features_r = torch.cat((image_features_r, image_features[-1:]), dim=0)

            logits = self.logit_scale * image_features_r @ self.text_features.t()

            marginal_prob = F.softmax(logits, dim=1).mean(0)

            pred_class = marginal_prob.argmax().item()
        
            # # Accuracy: 55.72%
            # # Latency: 234.35 ms

            ####################################################  
            # # # ll = []

            # # # for cluster_idx in range(6):
            # # #     cluster_indices = (cluster_labels == cluster_idx)
            # # #     if len(cluster_indices) == 0:
            # # #         continue

            # # #     cluster_features = image_features[cluster_indices]
            # # #     logits = self.logit_scale * cluster_features @ self.text_features.t()
            # # #     logits = logits.mean(dim=0)



            # # #     ll.append(logits)
            # # #     print(logits.shape)
            # # # # ll
            # # # # print(ll.shape)
            # # # ll = torch.stack(ll, dim=0)
            # # # ll = ll.mean(dim=0, keepdim=True)

            # # # marginal_prob = F.softmax(ll, dim=1).mean(0)
            # # # pred_class = marginal_prob.argmax().item()
            ####################################################  

            # exit()
            
            # image_features = torch.cat((image_features_r, image_features[-1:]), dim=0)
        
        
        
            # logits = self.logit_scale * image_features @ self.text_features.t()
            # marginal_prob = F.softmax(logits, dim=1).mean(0)
            # pred_class = marginal_prob.argmax().item()

            # # Visualize each cluster's images
            # import matplotlib.pyplot as plt
            # from torchvision.transforms.functional import to_pil_image
            
            # for cluster_idx in range(6):
            #     # Get indices of images in this cluster
            #     cluster_indices = (cluster_labels == cluster_idx).nonzero()[0]
                
            #     if len(cluster_indices) == 0:
            #         continue
                    
            #     print(f"Cluster {cluster_idx} has {len(cluster_indices)} images")
                
            #     # Setup plot
            #     cols = min(8, len(cluster_indices))
            #     rows = (len(cluster_indices) + cols - 1) // cols
            #     plt.figure(figsize=(cols * 2, rows * 2))
            #     plt.suptitle(f"Cluster {cluster_idx} - {len(cluster_indices)} images")
                
            #     for plot_idx, img_idx in enumerate(cluster_indices, start=1):
            #         if plot_idx > cols * rows:
            #             break
                        
            #         img = x[img_idx].permute(1, 2, 0).cpu().numpy()
            #         img = to_pil_image(img)
            #         plt.subplot(rows, cols, plot_idx)
            #         plt.imshow(img)
            #         plt.axis('off')
                
            #     plt.tight_layout()
            #     plt.show()

            ####################################################################


            # # # Visualize cluster_labels
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 8))
            # plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
            # plt.title('KMeans Clustering of Image Features')
            # plt.xlabel('Feature 1')
            # plt.ylabel('Feature 2')
            # plt.colorbar()
            # plt.show()

            # from sklearn.manifold import TSNE

            # # Apply t-SNE
            # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            # X_tsne = tsne.fit_transform(X_scaled)
            
            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(10, 8))

            # # If you did clustering, color by cluster
            # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.6)

            # # If you have true labels, you could color by those instead
            # # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_labels, cmap='viridis', alpha=0.6)

            # plt.colorbar(scatter)
            # plt.title('t-SNE Visualization with Clusters')
            # plt.xlabel('t-SNE dimension 1')
            # plt.ylabel('t-SNE dimension 2')
            # plt.show()
            
            
            # cosine_sim = torch.mm(image_features, image_features.t())
            # cosine_sim = image_features @ image_features.t()

            # # kmeans clusters
            # kmeans = KMeans(n_clusters=8, random_state=42)
            # kmeans.fit(cosine_sim.cpu().numpy())
            # print(kmeans.labels_)

            # show clusters




            # #  2. Get similarities of the last image ([-1]) with all others
            # last_img_similarities = cosine_sim[-1, :]  # (B,)

            # # 3. Sort indices (descending order, excluding the last image itself)
            # sorted_indices = torch.argsort(last_img_similarities, descending=True).cpu().numpy()
            # # sorted_indices = sorted_indices[sorted_indices != len(x)-1]  # Remove self-comparison

            # # 4. Visualize the last image + top-k most similar images
            # k = 63  # Number of similar images to display
            # cols = 8
            # rows = (image_features.shape[0] + cols - 1) // cols
            # plt.figure(figsize=(cols * 2, rows * 2))
            # for i, idx in enumerate(sorted_indices[:k], start=2):
            #     img = x[idx].permute(1, 2, 0).cpu().numpy()
            #     img = TF.to_pil_image(img)

            #     plt.subplot(rows, cols, i)
            #     plt.imshow(img)
            #     plt.title(f"Sim: {last_img_similarities[idx]:.3f}")
            #     plt.axis('off')

            # plt.tight_layout()
            # plt.show()


            # Perform KMeans clustering
            # kmeans = KMeans(n_clusters=4, random_state=42)
            # kmeans.fit(cosine_sim.cpu().numpy())
            # labels = kmeans.labels_
            # print(labels)

            # exit()
    
        return int(pred_class)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        custom_transform=kornia_random_crop,
        n_views=63,
    )

    dataloader, dataset = ResnetA(augmenter)

    # Load the CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(
        # model_name="ViT-B-32", pretrained="datacomp_xl_s13b_b90k", device=device#, force_quick_gelu=True
        model_name="ViT-B-16",
        pretrained="openai",
        device=device,
        force_quick_gelu=True,
    )
    clip_model.eval()

    # Create a ClipSkeleton instance
    wrapper_clip = ClipWrapper(
        clip_model, class_labels=dataset.class_code_to_label, device=device
    ).to(device)

    bench(wrapper_clip, dataloader, device, reduce=200, comment="", visualize=False)
