import os
import yaml



def generate_cent_yaml(data_path, output_dir, yaml_path, avg_lora=True, lora_enable=True, avg_projector=False, tune_mm_mlp_adapter=False):
    config = {
        "fed_alg": "local_fedavg_0",
        "lora_enable": lora_enable,
        "tune_mm_mlp_adapter": tune_mm_mlp_adapter,
        "test_local": True,
        "mm_projector_type": "mlp2x_gelu",
        "mm_vision_select_layer": -2,
        "mm_use_im_start_end": False,
        "mm_use_im_patch_token": False,
        "seed": 2025,
        "output_dir": output_dir,
        "dataset_name": "4task",
        "split_strategy": "cent",
        "data_path": data_path,
        "image_folder": "/grp01/saas_lqqu/wy/data/Multi_task/data2/image/",
        "model_name_or_path": "/grp01/saas_lqqu/wy/pretrain_model/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72",
        "llm_type": "meta-llama",
        "use_peft": True,
        "peft_lora_r": 8,
        "peft_lora_alpha": 32,
        "vision_tower_type": "clip",
        "vision_tower": "/grp01/saas_lqqu/wy/pretrain_model/models--openai--clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "pad_token_version": "v1",
        "image_aspect_ratio": "pad",
        "group_by_modality_length": True,
        "lr_scheduler_type": "cosine",
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "num_rounds": 20,
        "batch_size": 8,
        "model_max_length": 512,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.0,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "save_steps": 100,
        "save_model_freq": 1,
        "save_total_limit": 1,
        "logging_steps": 10,
        "gradient_checkpointing": True,
        "report_to": "tensorboard",
        "template": "alpaca",
        "bf16": True,
        "tf32": True
    }
    if not os.path.exists(os.path.dirname(yaml_path)):
        os.makedirs(os.path.dirname(yaml_path))

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def cent_yaml():

    # client_mixtask_per_client
    generate_cent_yaml(
        data_path=f"./data/Fed-Nature/",
        output_dir=f"./output/Fed-Nature/encoder_based/cent/pro/",
        avg_lora=False, 
        lora_enable=False, 
        avg_projector=False, 
        tune_mm_mlp_adapter=True,
        yaml_path=f"./Nature_Multi/config_encoder_based/cent_auto/pro_config.yaml"

    )
    generate_cent_yaml(
        data_path=f"./data/Fed-Nature/",
        output_dir=f"./output/Fed-Nature/encoder_based/cent/lora/",
        avg_lora=False, 
        lora_enable=True, 
        avg_projector=False, 
        tune_mm_mlp_adapter=False,
        yaml_path=f"./Nature_Multi/config_encoder_based/cent_auto/lora_config.yaml"
    )
    generate_cent_yaml(
        data_path=f"./data/Fed-Nature/",
        output_dir=f"./output/Fed-Nature/encoder_based/cent/pro_lora/",
        avg_lora=False, 
        lora_enable=True, 
        avg_projector=False, 
        tune_mm_mlp_adapter=True,
        yaml_path=f"./Nature_Multi/config_encoder_based/cent_auto/pro_lora_config.yaml"
    )



def generate_fed_yaml(fed_alg,task_type,task_name,data_path, output_dir, yaml_path, avg_lora=True, lora_enable=True, avg_projector=False, tune_mm_mlp_adapter=False):
    config = {
        "fed_alg": fed_alg,
        "avg_lora": avg_lora,
        "lora_enable": lora_enable,
        "avg_projector": avg_projector,
        "tune_mm_mlp_adapter": tune_mm_mlp_adapter,
        "test_local": True,
        "mm_projector_type": "mlp2x_gelu",
        "mm_vision_select_layer": -2,
        "mm_use_im_start_end": False,
        "mm_use_im_patch_token": False,
        "seed": 2025,
        "output_dir": output_dir,
        "dataset_name": task_name,
        "split_strategy": task_type,
        "data_path": data_path,
        "image_folder": "./data/Multi_task/data2/image/",
        "model_name_or_path": "./pretrain_model/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72",
        "llm_type": "meta-llama",
        "use_peft": True,
        "peft_lora_r": 8,
        "peft_lora_alpha": 32,
        "vision_tower_type": "clip",
        "vision_tower": "./pretrain_model/models--openai--clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "pad_token_version": "v1",
        "image_aspect_ratio": "pad",
        "group_by_modality_length": True,
        "lr_scheduler_type": "cosine",
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "num_rounds": 50,
        "batch_size": 8,
        "model_max_length": 512,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.0,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "save_steps": 100,
        "save_model_freq": 1,
        "save_total_limit": 1,
        "logging_steps": 20,
        "gradient_checkpointing": True,
        "report_to": "tensorboard",
        "template": "alpaca",
        "bf16": True,
        "tf32": True
    }
    import os
    import yaml
    if not os.path.exists(os.path.dirname(yaml_path)):
        os.makedirs(os.path.dirname(yaml_path))

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
  


def fed_yaml(method):
    generate_fed_yaml(
        fed_alg=method,
        task_type=f"pro",
        data_path=f"./data/Fed-Nature/",
        output_dir=f"./output/Fed-Nature/encoder_based/fed/{method}/pro/",
        avg_lora=False, 
        lora_enable=False, 
        avg_projector=True, 
        tune_mm_mlp_adapter=True,
        task_name="Fed-Nature",
        yaml_path=f"./Nature_Multi/config_encoder_based/{method}/pro_config.yaml"
    )
    generate_fed_yaml(
        fed_alg=method,
        task_type=f"lora",
        data_path=f"./data/Fed-Nature/",
        output_dir=f"./output/Fed-Nature/encoder_based/fed/{method}/lora/",
        avg_lora=True, 
        lora_enable=True, 
        avg_projector=False, 
        tune_mm_mlp_adapter=False,
        task_name="Fed-Nature",
        yaml_path=f"./Nature_Multi/config_encoder_based/{method}/lora_config.yaml"
    )
    generate_fed_yaml(
        fed_alg=method,
        task_type=f"pro_lora",
        data_path=f"./data/Fed-Nature/",
        output_dir=f"./output/Fed-Nature/encoder_based/fed/{method}/pro_lora/",
        avg_lora=True, 
        lora_enable=True, 
        avg_projector=True, 
        tune_mm_mlp_adapter=True,
        task_name="Fed-Nature",
        yaml_path=f"./Nature_Multi/config_encoder_based/{method}/pro_lora_config.yaml"
    )
    generate_fed_yaml(
        fed_alg=method,
        task_type=f"2stage",
        data_path=f"./data/Fed-Nature/",
        output_dir=f"./output/Fed-Nature/encoder_based/fed/{method}/2stage/",
        avg_lora=True, 
        lora_enable=True, 
        avg_projector=True, 
        tune_mm_mlp_adapter=True,
        task_name="Fed-Nature",
        yaml_path=f"./Nature_Multi/config_encoder_based/{method}/2stage_config.yaml"
    )


fed_yaml("fedavg")
fed_yaml("fedprox")
fed_yaml("fedyogi")
fed_yaml("fedadam")
fed_yaml("fedavgm")
fed_yaml("fedadagrad")
