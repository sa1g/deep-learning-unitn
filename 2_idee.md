Voglio migliorare CLIP senza rallentarlo troppo. Tendenzialmente fare image augmentation e' piu' rapido di fare backpropagation, quindi i metodi con backprop non li ho seguiti troppo, ma ho preso comunque spunto da alcuni.

1. come da (DINOv2: Learning Robust Visual Features without Supervision)[https://arxiv.org/pdf/2304.07193] prendiamo spunto dal `self-supervised image retrieval` non per costruire un dataset migliore, ma per filtrare le augmentation del TTA: sostituiamo il confidence selection con questo metodo.
    0. dal loro image-agumenter fanno
        1. random crop
            1. color jitter (brightness 04, contrast 0.4, saturation 0.2, hue 0.1), p=0.8
            2. randomgrayscale p=0.2
    
    1. calcolo image-embedding usanto ViT-B/16
    2. calcolo cosine-similarity tra immagini
    3. k-means clustering delle immagini
    4. prendo quelle vicine a quella iniziale (o a quelle con confidenza piu' alta)

    > Self-supervised image retrieval. We build our curated pretraining dataset by retrieving images from our uncurated data source that are close to images in our curated sources. In order to do this, we first compute an image embedding using a self-supervised ViT-H/16 network pretrained on ImageNet-22k, and use cosine-similarity as a distance measure between images. Then, we perform k-means clustering of the uncurated data. Given a query dataset for retrieval, if it is large enough we retrieve N (typically 4) nearest neighbors for each query image. If it is small, we sample M images from the cluster corresponding to each query image. Although visual inspection seemed to indicate good retrieval quality for N much larger than 4, this leads to more collisions (images that are nearest-neighbor retrievals of multiple queries). We choose N = 4 as it provides a good tradeoff in that sense

    - Sempre prendendo spunto da DINOv2 sarebbe figo far vedere che l'embedding di ViT-B/16 non e' come quello di DINOv2 dove e' "facile" vedere la _saliency map_ delle immagini, quindi le zone "piu' importanti" per estrarre le label. Pero' su una bella % di immagini la cosa funziona piuttosto bene. Da notare che mascherando le immagini con la saliency map per forzare CLIP (ViT-B/16) a interessarsi solo alle parti _salienti_ non e' utile visto che al massimo performa come da default.
    - Se si potesse fare training (di piccoli pezzi, su terzi dataset) sarebbe figo fare sia il 2. che (Convolutional Visual Prompt for Robust Visual Perception)[https://arxiv.org/pdf/2303.00198], che potrebbe aiutare in TTA.

2. (Efficient Test-Time Adaptation of Vision-Language Models)[https://arxiv.org/pdf/2403.18293] sembra essere TPT con un "dynamic adapter", aka key-value storage tra prompt e image-encoder. Leggere come funziona. E' training-free, tempo che funzioni con few-shots.

3. (Noise is an Efficient Learner for Zero-Shot Vision-Language Models)[https://arxiv.org/pdf/2502.06019]: TNT, funziona su single-image w/augmentations, bisogna fare piu' epoche, ma pare funzionare bene. Dovrebbe essere compatibile con TPT. Learnable _adaptive noise $\mathcal{E}$



- EXTRA: (Test-Time Low Rank Adaptation via Confidence Maximization for Zero-Shot Generalization of Vision-Language Models)[https://arxiv.org/pdf/2407.15913]: TTL, molto figo, LORA (low rank adapters) per il visual encoder, senza TPT. Backprop (w/AdamW), lo sento "pesante". Se e' veloce da implementare proviamo.
- EXTRA: (RLCF)[https://arxiv.org/pdf/2305.18010]: pesante, di massima richiede un LLM.
