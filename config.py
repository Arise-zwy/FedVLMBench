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
import glob

# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    save_model_freq: Optional[int] = field(default=50, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})
    avg_lora: Optional[bool] = field(default=False, metadata={"help": "whether to average the LoRA parameters"})
    avg_projector: Optional[bool] = field(default=False, metadata={"help": "whether to average the projector parameters"})
    test_local: Optional[bool] = field(default=False, metadata={"help": "whether to test the local model"})

## parameters of the mobilevlm model
@dataclass
class ModelArguments:
    llm_type: Optional[str] = field(default="")

    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pad_token_version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)

    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_type: Optional[str] = field(default='clip')
    lora_path: Optional[str] = field(default=None, metadata={"help": "the path to the LoRA parameters"})


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
    dataset_sample: Optional[int] = field(default=20000, metadata={"help": "the number of samples to use from the dataset"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})
    config: Optional[str] = field(default=None, metadata={"help": "the config file"})

   


@dataclass
class ScriptArguments(transformers.TrainingArguments):
    resume_from_checkpoint: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
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
    lora_enable: bool = False
    # lora_r: int = 64
    # lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})    # vicuna and alpaca use 2e-5
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    # eval_batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size of evaluation"})
    # eval_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of accumulation steps of evaluation"})
    # seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})

    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    # max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})

    should_save: Optional[bool] = field(default=True, metadata={"help": "whether to save the model"})
    
def convert_value(value):
    if value is None or value == []:
        return None

    if isinstance(value, bool):
        return value
    try:
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
        return float_value
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

def get_config(config):
    import yaml
    # 读取 YAML 配置文件
    with open(config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    parser = HfArgumentParser((ScriptArguments, FedArguments,ModelArguments, DataArguments))
    script_args, fed_args, model_args, data_args = parser.parse_args_into_dataclasses()

    args_mapping = {
        ScriptArguments: script_args,
        FedArguments: fed_args,
        ModelArguments: model_args,
        DataArguments: data_args
    }
    update_args_from_config(args_mapping, config)
    # ===== Define the LoraConfig =====
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None
    return script_args, fed_args, peft_config, model_args, data_args

def get_testing_config(config):
    import yaml
    # 读取 YAML 配置文件
    with open(config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    parser = HfArgumentParser((ScriptArguments,ModelArguments, DataArguments))
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()

    args_mapping = {
        ScriptArguments: script_args,
        ModelArguments: model_args,
        DataArguments: data_args
    }
    print(config)
    update_args_from_config(args_mapping, config)

    return script_args, model_args, data_args


def get_testing_config_ori_dir(json_path):
    # 读取 JSON 配置文件
    with open(json_path, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as exc:
            print(exc)
            return None, None, None

    parser = HfArgumentParser((ScriptArguments, ModelArguments, DataArguments))
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()

    args_mapping = {
        ScriptArguments: script_args,
        ModelArguments: model_args,
        DataArguments: data_args
    }

    def flatten_dict(d, sep='_'):
        items = []
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(flatten_dict(v, k).items())
            else:
                items.append((k, v))
        return dict(items)

    config = flatten_dict(config)
    print(config)
    
    update_args_from_config(args_mapping, config)
 

    return script_args, model_args, data_args


# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = ScriptArguments(
    # training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        # max_steps=script_args.max_steps,
        report_to=script_args.report_to,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        logging_dir=script_args.logging_dir,
        lr_scheduler_type="constant",
        fp16=True,
        evaluation_strategy="no",
        mm_projector_lr=script_args.mm_projector_lr,
        weight_decay=script_args.weight_decay,
        group_by_modality_length=script_args.group_by_modality_length,
        resume_from_checkpoint=script_args.resume_from_checkpoint,
        batch_size=script_args.batch_size,
        disable_tqdm=True,
        dataloader_num_workers=4,
        # eval_batch_size=4,
        # eval_accumulation_steps=4,
    )
    return training_args


# ===== Define the training arguments =====
def get_training_config(config, new_lr):
    training_args = ScriptArguments(
    # training_args = TrainingArguments(
        output_dir=config.experiment.output_dir,
        per_device_train_batch_size=config.training.batch_size_mmu,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=config.experiment.logging_steps,
        num_train_epochs=config.fed.num_train_epochs,
        # max_steps=config.fed.max_steps,
        max_steps=30,
        should_save=False,


        report_to=config.fed.report_to,
        # save_steps=config.experiment.save_steps,
        save_steps=500,
        save_total_limit=config.experiment.save_total_limit,
        # push_to_hub=config.fed.push_to_hub,
        # hub_model_id=config.fed.hub_model_id,
        gradient_checkpointing=config.model.gradient_checkpointing,
        logging_dir=config.experiment.output_dir+"/logs",
        lr_scheduler_type="constant",
        fp16=True,
        evaluation_strategy="no",
        # mm_projector_lr=config.fed.mm_projector_lr,
        weight_decay=config.optimizer.params.weight_decay,
        group_by_modality_length=config.fed.group_by_modality_length,
        # resume_from_checkpoint=config.fed.resume_from_checkpoint,
        batch_size=config.training.batch_size_mmu,
        disable_tqdm=True,
        dataloader_num_workers=4,
        # eval_batch_size=4,
        # eval_accumulation_steps=4,
    )
    return training_args

# ===== Define the testing arguments =====
def get_testing_args(script_args):
    testing_args = ScriptArguments(
        output_dir=script_args.output_dir,
        logging_steps=script_args.logging_steps,
        fp16=True,
      
    )
    return testing_args


def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    elif script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    return device_map, quantization_config, torch_dtype

def save_config(script_args, fed_args, model_args, data_args):
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    dataset_name_split = os.path.basename(data_args.dataset_name)
    
    client_files = glob.glob(data_args.data_path + "train/client_*.json")
    fed_args.num_clients = len(client_files)
    fed_args.sample_clients = len(client_files)
    # output_dir = f"{script_args.output_dir}/{dataset_name_split}_{data_args.dataset_sample}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.model_max_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.split_strategy}_{fed_args.fed_alg}_{model_args.mm_projector_type}_c{fed_args.num_clients}s{fed_args.sample_clients}_L{script_args.learning_rate}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.model_max_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    logging_dir= output_dir+"/logs"
    while True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.fed_alg}_{model_args.mm_projector_type}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.model_max_length}_{now_time}"
    if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

    script_args.output_dir = output_dir
    script_args.logging_dir=logging_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "fed_args": asdict(fed_args),
            "model_args": asdict(model_args),
            "script_args": asdict(script_args),
            "data_args": asdict(data_args),
        }
        json.dump(combined_dict, f, indent=4)
    return script_args