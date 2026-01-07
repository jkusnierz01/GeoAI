# EuroSAT:
## Overall performance:
| Metric | With Embeddings | Without Embeddings | Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **92.74%** | 90.87% | +1.87% |
| **Test Loss** | **0.2278** | 0.2748 | -0.0470 |
| **Weighted F1**| **0.9274** | 0.9063 | +0.0211 |

## Class-Specific
| Class | F1 (With Embeddings) | F1 (No Embeddings) | Impact |
| :--- | :--- | :--- | :--- |
| **AnnualCrop** | **0.9248** | 0.8935 | +0.0313 |
| **Forest** | 0.9299 | **0.9456** | -0.0157 |
| **HerbaceousVegetation** | **0.9253** | 0.8756 | +0.0497 |
| **Highway** | **0.9064** | 0.8907 | +0.0157 |
| **Industrial** | **0.9571** | 0.9387 | +0.0184 |
| **Pasture** | 0.8819 | **0.9066** | -0.0247 |
| **PermanentCrop** | **0.8692** | 0.7380 | +0.1312 |
| **Residential** | **0.9817** | 0.9764 | +0.0053 |
| **River** | **0.9116** | 0.9043 | +0.0073 |
| **SeaLake** | 0.9597 | **0.9681** | -0.0084 |