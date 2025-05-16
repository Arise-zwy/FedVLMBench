# FedVLMBench: Benchmarking Federated Fine-Tuning of Vision Language Models

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Downloading](#Downloading)
- [Running Experiments](#running-experiments)
- [License](#license)

## Prerequisites

- Linux operating system
- NVIDIA GPU with CUDA capability
- Conda package manager
- Python 3.8+

## Installation

### 1. Set Up Conda Environment

```
conda env create -f env.yaml
conda activate fedvlm
```

### 2. Configuration
#### Code
```
/OpenFedLLM-main
├── /Nature_Multi
│   ├── main_encoder_based_fed_natural.py
│   ├── main_encoder_free_fed_natural.py
│   ├── main_encoder_based_local_natural.py
│   └── main_encoder_free_local_natural.py
├── /FGVC
│   ├── main_encoder_based_fed_fgvc.py
│   ├── main_encoder_free_fed_fgvc.py
│   ├── main_encoder_based_local_fgvc.py
│   └── main_encoder_free_local_fgvc.py
└── /gen_config
    ├── /encoder_free
    │   ├── gen_Natural_Multi_config.py
    │   └── gen_FGVC_config.py
    └── /encoder_based
        ├── gen_FGVC_config.py
        └── gen_Nature_Multi_config.py
```

#### Fed-FGVC
```
/Fed-FGVC
├── /clients
│   ├── /train
│   └── /test
└── /central_training
    ├── /train
    └── /test
```

#### Fed-Med
```
/Fed-Med
├── /image
│   ├── /MIMIC-CXR
│   ├── /RadGnome
│   ├── /slake
│   └── /RAD-VQA
├── /clients
│   ├── /train
│   └── /test
└── /central_training
    ├── /train
    └── /test
```

#### Fed-RadGenome
```
/Fed-RadGenome
├── /clients
│   ├── /train
│   └── /test
└── /central_training
    ├── /train
    └── /test
```

#### Fed-Nature
```
/Fed-Nature
├── /clients
│   ├── /train
│   └── /test
└── /central_training
    ├── /train
    └── /test
```


#### Fed-ScienceCap
```
/Fed-ScienceCap
├── /clients
│   ├── /train
│   └── /test
└── /central_training
    ├── /train
    └── /test
```

### 3. Downloading

#### Step1: Download Models
First, Download the models in the ./pretrained_models 
##### LLaMA-Instruct 3.2 3B
```
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

##### CLIP ViT-B-32

```
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
```
##### Show-o 1.5B
https://huggingface.co/showlab/show-o-512x512/tree/main

#### Step2: Download Images
Download the images in the ./images

##### FGVC
https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

##### COCO
https://cocodataset.org/#home

##### RefCOCO
https://github.com/lichengunc/refer

##### COCO-QA
https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/

##### ScienceQA
https://scienceqa.github.io/

##### RadGenome-ChestCT
https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT

##### VQA-RAD
https://huggingface.co/datasets/flaviagiammarino/vqa-rad

##### MIMIC-CXR
https://physionet.org/content/mimic-cxr/2.1.0/

##### SLAKE
https://www.med-vqa.com/slake/


### 4. Running Experiments

#### Step1: Generate Configuration Files
Generate your configuration files before running experiments:
```
# Generate Natural_Multi configs
python gen_config/encoder_free/gen_Natural_Multi_config.py
python gen_config/encoder_based/gen_Natural_Multi_config.py

# Generate FGVC configs  
python gen_config/encoder_free/gen_FGVC_config.py
python gen_config/encoder_based/gen_FGVC_config.py
```


#### Step2: Run Training Experiments
```
# Federated training (encoder_free VLM)
python Nature_Multi/main_encoder_free_fed_natural.py --config_path ./Nature_Multi/encoder_free/fedavg_auto/config.yaml

# Federated training (encoder-based VLM) 
python Nature_Multi/main_encoder_based_fed_natural.py --config_path ./Nature_Multi/encoder_based/fedavg_auto/config.yaml


# Federated training (encoder_free VLM)
python Nature_Multi/main_encoder_free_local_natural.py --config_path ./Nature_Multi/encoder_free/fedavg_auto/config.yaml

# Central training (encoder-based VLM)
python Nature_Multi/main_encoder_based_fed_natural.py --config_path ./Nature_Multi/encoder_based/cent_auto/pro_lora_config.yaml

```


