import os
import yaml


def generate_cent_yaml(data_path, output_dir, yaml_path, avg_lora=True, lora_enable=True, avg_projector=False, tune_mm_mlp_adapter=False):
    config = {
        "fed_alg": "local_fedavg_0",
        "lora_enable": lora_enable,
        "tune_mm_mlp_adapter": tune_mm_mlp_adapter,
        "test_local": False,
        "mm_projector_type": "mlp6x_gelu",
        "mm_vision_select_layer": -2,
        "mm_use_im_start_end": False,
        "mm_use_im_patch_token": False,
        "seed": 2025,
        "output_dir": output_dir,
        "dataset_name": "medvqa",
        "split_strategy": "cent",
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


def generate_fed_yaml(pro_type,fed_alg,task_type,task_name,data_path, output_dir, yaml_path, avg_lora=True, lora_enable=True, avg_projector=False, tune_mm_mlp_adapter=False):
    if pro_type == "linear":
        mm_projector_type = "linear"
    elif pro_type == "mlp6x":
        mm_projector_type = "mlp6x_gelu"
    elif pro_type == "mlp2x":
        mm_projector_type = "mlp2x_gelu"
    else:
        mm_projector_type = "mlp2x_gelu"

    config = {
        "fed_alg": fed_alg,
        "avg_lora": avg_lora,
        "lora_enable": lora_enable,
        "avg_projector": avg_projector,
        "tune_mm_mlp_adapter": tune_mm_mlp_adapter,
        "test_local": False,
        "mm_projector_type": mm_projector_type,
        "mm_vision_select_layer": -2,
        "mm_use_im_start_end": False,
        "mm_use_im_patch_token": False,
        "seed": 2025,
        "output_dir": output_dir,
        "dataset_name": task_name,
        "split_strategy": task_type,
        "data_path": data_path,
        "image_folder": "./data/Fed-FGVC/image/",
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
        "model_max_length": 1024,
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
        for data_split in ["iid", "non_iid"]:
            generate_fed_yaml(
                fed_alg=method,
                task_type=f"pro_lora",
                data_path= f"./data/Fed-FGVC/",
                output_dir=f"./output/Fed-FGVC/encoder_based/fed/pro_lora/{data_split}/{method}/",
                avg_lora=True, 
                lora_enable=True, 
                avg_projector=True, 
                tune_mm_mlp_adapter=True,
                task_name="Fed-FGVC",
                yaml_path=f"./FGVC/config_encoder_based/fed/{data_split}/{method}/pro_lora_config.yaml"

            )
            generate_fed_yaml(
                fed_alg=method,
                task_type=f"pro",
                data_path= f"./data/Fed-FGVC/",
                output_dir=f"./output/Fed-FGVC/encoder_based/fed/pro/{data_split}/{method}/",
                avg_lora=False, 
                lora_enable=False, 
                avg_projector=True, 
                tune_mm_mlp_adapter=True,
                task_name="Fed-FGVC",
                yaml_path=f"./FGVC/config_encoder_based/fed/{data_split}/{method}/pro_config.yaml"
            )
            generate_fed_yaml(
                fed_alg=method,
                task_type=f"lora",
                data_path= f"./data/Fed-FGVC/",
                output_dir=f"./output/Fed-FGVC/encoder_based/fed/lora/{data_split}/{method}/",
                avg_lora=True, 
                lora_enable=True, 
                avg_projector=False, 
                tune_mm_mlp_adapter=False,
                task_name="Fed-FGVC",
                yaml_path=f"./FGVC/config_encoder_based/fed/{data_split}/{method}/lora_config.yaml"
            )
            generate_fed_yaml(
                fed_alg=method,
                task_type=f"2stage",
                data_path= f"./data/Fed-FGVC/",
                output_dir=f"./output/Fed-FGVC/encoder_based/fed/pro_lora/{data_split}/{method}/",
                avg_lora=True, 
                lora_enable=True, 
                avg_projector=False, 
                tune_mm_mlp_adapter=False,
                task_name="Fed-FGVC",
                yaml_path=f"./FGVC/config_encoder_based/fed/{data_split}/{method}/2stage_config.yaml"
            )
              



fed_yaml("fedavg")
fed_yaml("fedprox")
fed_yaml("fedyogi")
fed_yaml("fedadam")
fed_yaml("fedavgm")
fed_yaml("fedadagrad")

