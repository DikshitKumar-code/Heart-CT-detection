---
title: Heart CT Segmentation
emoji: ❤️‍🩹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Heart CT Scan Segmentation Demo

This is a live application for segmenting the heart from CT scan images.

**Try it live here:**  
[👉 Heart CT Segmentation Demo on Hugging Face Spaces](https://huggingface.co/spaces/Dk123gfdhs/heart-segmentation-demo)

You can choose from three different models to perform the segmentation:
*   **Custom UNet**: A standard UNet model built from scratch.
*   **SMP UNet (EfficientNet-B7)**: A powerful, pre-trained UNet from the `segmentation-models-pytorch` library.
*   **SegFormer**: A modern transformer-based model for semantic segmentation.

Upload an image and select a model to see the predicted mask.


## Model Performance Metrics

Below are the evaluation results for each of the models on the test set.

### SegFormer Model

| Metric             | Value    |
| :----------------- | :------- |
| Accuracy           | 0.9935   |
| Precision (Heart)  | 0.9001   |
| Recall (Heart)     | 0.8996   |
| F1 Score (Heart)   | 0.8999   |

### Custom UNet Model

| Metric         | Value    |
| :------------- | :------- |
| Test Loss      | 0.2742   |
| Test Accuracy  | 0.9909   |
| Test IoU Score | 0.7421   |

### Fine-tuned SMP UNet Model

| Metric         | Value    |
| :------------- | :------- |
| Test Loss      | 0.0943   |
| Test Accuracy  | 0.9962   |
| Test IoU Score | 0.8858   |
