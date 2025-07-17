---
title: Heart CT Segmentation
emoji: ❤️‍🩹
colorFrom: blue
colorTo: green
sdk: python
app_file: app.py
pinned: false
---

# Heart CT Scan Segmentation Demo

This is a live application for segmenting the heart from CT scan images.

You can choose from three different models to perform the segmentation:
*   **Custom UNet**: A standard UNet model built from scratch.
*   **SMP UNet (EfficientNet-B7)**: A powerful, pre-trained UNet from the `segmentation-models-pytorch` library.
*   **SegFormer**: A modern transformer-based model for semantic segmentation.

Upload an image and select a model to see the predicted mask.
