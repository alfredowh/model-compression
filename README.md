# Model Compression
Repository for model compression project using pruning and knowledge distillation methods.
- Pruning methods:

- Fine-Tuning methods: Retraining, Iterative Pruning, Knowledge Distillation
## Setup

1. Download the ImageNet dataset
2. Run training using `train.py`
3. Evaluate models using `test.py`

## Folder Structure

``` bash
.
├── data                        # ImageNet data 
├── models/                     # model architecture & pruning 
├── notebooks/                  # Exploration & eval notebook
├── runs/                       # Experiment results
├── utils/                      # Utility functions
├── train.py                    # Training script
└── test.py                     # Eval script
```

### Citations
[`TinyML and Efficient Deep Learning Computing`](https://efficientml.ai/)

```bash
@software{torchvision2016,
    title        = {TorchVision: PyTorchs Computer Vision library},
    author       = {TorchVision maintainers and contributors},
    year         = 2016,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/pytorch/vision}}
}
