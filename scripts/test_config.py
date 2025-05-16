from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime, timedelta
import transformers



@dataclass
class ModelArguments:
    llm_type: Optional[str] = field(default="")

    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pad_token_version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_type: Optional[str] = field(default='clip')


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    
    dataset_name: Optional[str] = field(
        default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"}
    )
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})



@dataclass
class ScriptArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
   
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
  
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
   
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})

    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    


def convert_value(value):
    if isinstance(value, bool):
        return value
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    return value

def update_args_from_config(args_mapping, config):
    for key, value in config.items():
        for args_class, args_instance in args_mapping.items():
            if hasattr(args_instance, key):
                setattr(args_instance, key, convert_value(value))

def get_testing_config(config):
    import yaml
    # 读取 YAML 配置文件
    with open(config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    parser = HfArgumentParser((ScriptArguments,ModelArguments, DataArguments))
    script_args, fed_args, model_args, data_args = parser.parse_args_into_dataclasses()

    args_mapping = {
        ScriptArguments: script_args,
        ModelArguments: model_args,
        DataArguments: data_args
    }
    update_args_from_config(args_mapping, config)

    return script_args, model_args, data_args

# ===== Define the training arguments =====
def get_testing_args(script_args, new_lr):
    testing_args = ScriptArguments(
        output_dir=script_args.output_dir,
        logging_steps=script_args.logging_steps,
        fp16=True,
        mm_projector_lr=script_args.mm_projector_lr,
        weight_decay=script_args.weight_decay,
        group_by_modality_length=script_args.group_by_modality_length,
      
    )
    return testing_args
