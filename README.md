# Plant Disease Classification

This project loads the CCMT dataset and keeps the train and test splits separate. It validates image files, applies resize/crop and ImageNet normalization, and then trains a MobileNet model using the original notebook logic.

## Implemented workflow
- Dataset loading for the CCMT folder structure with separate train/test sets
- Cleaning of class names by removing trailing numeric suffixes
- Image validation and filtering of bad files
- Data augmentation and preprocessing using resize, random crop, horizontal flip, and ImageNet normalization
- Model creation with `timm` and training using `CrossEntropyLoss`, `Adam`, and cosine annealing
- Weighted accuracy, precision, recall, and F1 evaluation
- Confusion matrix and classification report generation
- Training curve and t-SNE visualization outputs

## Important note
The original notebook code in this project currently creates the model with `timm.create_model('mobilenetv3_large_100', ...)` rather than the `MobileNetV4-Conv-Medium` name mentioned in the project notes. This module preserves the existing code exactly and does not modify the original behavior.
