# Wildfire-Mapper-Sen2Fire

A PyTorch implementation for wildfire semantic segmentation using the Sen2Fire benchmark dataset, based on the findings from the paper: [**SEN2FIRE: A Challenging Benchmark Dataset for Wildfire Detection Using Sentinel Data**](https://arxiv.org/abs/2403.17884).

This project explores how different combinations of satellite bands from Sentinel-2 and the aerosol index from Sentinel-5P can be used to accurately map active wildfires.

## Dataset

-   **Download Link:** [https://www.kaggle.com/datasets/shariaarfin/sen2fire?resource=download&select=scene4]([https://zenodo.org/records/10881058](https://www.kaggle.com/datasets/shariaarfin/sen2fire?resource=download&select=scene4))
-   **Structure:** The dataset is split into `train`, `validation`, and `test` sets from four distinct geographical regions in Australia to ensure model generalization.
-   **Format:** Each sample is a `.npz` file containing a `(12, 512, 512)` image (Sentinel-2 bands), a `(512, 512)` aerosol channel (Sentinel-5P), and a `(512, 512)` binary label mask.
