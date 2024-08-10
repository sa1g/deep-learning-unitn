import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def visualise_data(images_path, texts):
    plt.figure(figsize=(16, 5))

    for i, (image_path, text) in enumerate(zip(images_path, texts)):
        # Load and visualize the image along with its text description
        image = Image.open(image_path).convert("RGB")

        plt.subplot(2, 4, i+1)
        plt.imshow(image)
        plt.title(text)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()


def visualise_similarity(similarity, images_path, texts):
    # Flip similarity (just for visualization)
    similarity = similarity.permute(1, 0)

    similarity = similarity.numpy()
    count = len(texts)

    plt.figure(figsize=(18, 12))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])

    for i, image_path in enumerate(images_path):
        image = Image.open(image_path).convert("RGB")
        plt.imshow(image, extent=(
            i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    # Print the scores
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}",
                     ha="center", va="center", size=12)

    # Update spines
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    # Change plot limits
    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    # Set title
    plt.title("Cosine similarity between text and image features", size=20)


def cosine_similarity(images_z, texts_z):
    # Normalise the image and the text
    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    # Evaluate the cosine similarity between the sets of features
    similarity = (images_z @ texts_z.T)

    return similarity.cpu()


def visualise_probabilities(images_path, classnames, texts_p, k=5):
    topk_p, topk_labels = texts_p.cpu().topk(k, dim=-1)
    plt.figure(figsize=(12, 12))

    for i, image_path in enumerate(images_path):
        # Read the image
        image = Image.open(image_path).convert("RGB")

        # Visualise the image
        plt.subplot(4, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        # Visualise the probabilities for the image
        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(topk_p.shape[-1])
        plt.grid()
        plt.barh(y, topk_p[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [classnames[index] for index in topk_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.show()
