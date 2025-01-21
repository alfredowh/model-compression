# Model Compression
Repository for model compression project using pruning and knowledge distillation methods.
- Pruning methods:
<p align="center">
    <img src="assets/pruning_methods.png" alt="img" width="540"/>
</p>

- Fine-Tuning methods: `Retraining`, `Iterative Pruning`, `Knowledge Distillation` 
## Setup

1. Download the ImageNet dataset
2. Visualization of model exploration in `notebooks/model_exploration.ipynb`
3. Run training using `train.py`
4. Evaluate models using `test.py`
5. Visualization of model evaluation in `notebooks/evaluation.ipynb`

## Folder Structure

``` bash
.
├── data/                       # ImageNet data, hyperparameter .yaml
├── models/                     # model architecture & pruning 
├── notebooks/                  # Exploration & eval notebook
├── runs/                       # Experiment results
├── scripts/                    # Example of sh scripts to run training & test
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
