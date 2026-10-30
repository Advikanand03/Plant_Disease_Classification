# Plant Disease Classification

This project loads the CCMT dataset and keeps the train and test splits separate. It validates image files, applies resize/crop and ImageNet normalization, and trains MobileNetV4-Conv-Medium using the notebook settings.

## Implemented workflow
- Dataset loading for the CCMT folder structure with separate train/test sets
- Cleaning of class names by removing trailing numeric suffixes
- Image validation and filtering of bad files
- Data augmentation and preprocessing using resize, random crop, horizontal flip, and ImageNet normalization
- Model creation with `timm` (`mobilenetv4_conv_medium.e500_r224_in1k`) and training using `CrossEntropyLoss`, `Adam`, and cosine annealing
- Data loaders use two workers, matching the notebook
- Weighted accuracy, precision, recall, and F1 evaluation
- Confusion matrix and classification report generation
- Training curve and t-SNE visualization outputs

## Dataset and run
The default dataset path in `main.py` is configured for Kaggle. Update `base_dir` to the local CCMT dataset location before running elsewhere. Install dependencies with `pip install -r requirements.txt`, then run `python main.py`.
