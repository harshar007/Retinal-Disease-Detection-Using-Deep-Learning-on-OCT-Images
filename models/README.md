# Models Directory

This directory stores the trained PyTorch state dictionaries (`.pt`) and serialized production model packages (`.pkl`).

### Note on GitHub File Size Limits
Due to GitHub's file size limits (files > 100MB cannot be pushed over standard Git), binary model weight files (`*.pt`, `*.pkl`) are excluded from version control via `.gitignore`. The best model package (`vgg16.pkl`) is approximately **1.07 GB**.

### How to Generate the Models
To train all architectures and generate the `.pkl` and `.pt` model files locally:

```bash
# Run the complete end-to-end training and evaluation pipeline
python run_pipeline.py
```

This will automatically populate this directory with:
- `best_retinal_oct_model.pkl` & `best_retinal_oct_model.pt`
- `vgg16.pkl` & `vgg16.pt`
- `densenet121.pkl` & `densenet121.pt`
- `mobilenet_v2.pkl` & `mobilenet_v2.pt`
- `resnet18.pkl` & `resnet18.pt`
- `custom_cnn.pkl` & `custom_cnn.pt`
- `evaluation_metrics.pkl`
