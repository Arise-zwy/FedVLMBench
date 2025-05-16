import os
import yaml



def generate_cent_yaml(data_path, output_dir, yaml_path):
    config = {
        "wandb": {
            "entity": None,
            "resume": "auto"
        },
        "experiment": {
            "project": "tuning",
            "name": "show-o-tuning-stage2",
            "output_dir": output_dir,
            "max_train_examples_mmu": 40000000,
            "save_every": 10000,
            "eval_every": 2500,
            "generate_every": 1000,
            "log_grad_norm_every": 500,
            "resume_from_checkpoint": "latest",
            "save_steps": 100,
            "save_total_limit": 1,
            "logging_steps": 10,
            "save_model_freq": 5
        },
        "model": {
            "vq_model": {
                "type": "magvitv2",
                "vq_model_name": "./pretrain_model/magvitv2"
            },
            "showo": {
                "load_from_showo": False,
                "pretrained_model_path": "./pretrain_model/showo",
                "w_clip_vit": False,
                "vocab_size": 58498,
                "llm_vocab_size": 50295,
                "llm_model_path": "microsoft/phi-1_5",
                "codebook_size": 8192,
                "num_vq_tokens": 256,
                "num_new_special_tokens": 10
            },
            "gradient_checkpointing": True
        },
        "dataset": {
            "dataset_name": "Fed-FGVC",
            "split_strategy": "cent",
            "data_path": data_path,
            "image_folder": "./data/Fed-FGVC/image/",
            "und_type": "llava_tuning",
            "add_system_prompt": False,
            "params": {
                "add_caption_prompt": True,
                "validation_prompts_file": "validation_prompts/text2image_prompts.txt",
                "shuffle_buffer_size": 1000,
                "num_workers": 32,
                "resolution": 256,
                "pin_memory": True,
                "persistent_workers": True
            },
            "preprocessing": {
                "max_seq_length": 512,
                "resolution": 256,
                "center_crop": False,
                "random_flip": False
            }
        },
        "optimizer": {
            "name": "adamw",
            "params": {
                "learning_rate": 5e-05,
                "scale_lr": False,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.01,
                "epsilon": 1e-8
            }
        },
        "lr_scheduler": {
            "scheduler": "cosine",
            "params": {
                "learning_rate": "${optimizer.params.learning_rate}",
                "warmup_steps": 1000
            }
        },
        "training": {
            "gradient_accumulation_steps": 1,
            "noise_type": "mask",
            "batch_size_mmu": 4,
            "mixed_precision": "bf16",
            "enable_tf32": True,
            "seed": 2025,
            "max_train_steps": 14000,
            "overfit_one_batch": False,
            "cond_dropout_prob": 0.1,
            "min_masking_rate": 0.0,
            "label_smoothing": 0.0,
            "max_grad_norm": None,
            "guidance_scale": 0.0,
            "generation_timesteps": 12,
            "mmu_coeff": 1.0
        },
        "fed": {
            "fed_alg": "fedavg",
            "test_local": False,
            "bits": 16,
            "image_aspect_ratio": "pad",
            "group_by_modality_length": True,
            "num_train_epochs": 1,
            "num_rounds": 50,
            "report_to": "tensorboard"
        }
    }
    if not os.path.exists(os.path.dirname(yaml_path)):
        os.makedirs(os.path.dirname(yaml_path))

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def generate_fed_yaml(fed_alg,split_type,data_name,data_path, image_path, output_dir, yaml_path):
   
    config = {
        "wandb": {
            "entity": None,
            "resume": "auto"
        },
        "experiment": {
            "project": "tuning",
            "name": "show-o-tuning-stage2",
            "output_dir": output_dir,
            "max_train_examples_mmu": 40000000,
            "save_every": 10000,
            "eval_every": 2500,
            "generate_every": 1000,
            "log_grad_norm_every": 500,
            "resume_from_checkpoint": "latest",
            "save_steps": 100,
            "save_total_limit": 1,
            "logging_steps": 10,
            "save_model_freq": 5
        },
        "model": {
            "vq_model": {
                "type": "magvitv2",
                "vq_model_name": "./pretrain_model/magvitv2"
            },
            "showo": {
                "load_from_showo": False,
                "pretrained_model_path": "./pretrain_model/showo",
                "w_clip_vit": False,
                "vocab_size": 58498,
                "llm_vocab_size": 50295,
                "llm_model_path": "microsoft/phi-1_5",
                "codebook_size": 8192,
                "num_vq_tokens": 256,
                "num_new_special_tokens": 10
            },
            "gradient_checkpointing": True
        },
        "dataset": {
            "dataset_name": data_name,
            "split_strategy": split_type,
            "data_path": data_path,
            "image_folder": image_path,
            "und_type": "llava_tuning",
            "add_system_prompt": False,
            "params": {
                "add_caption_prompt": True,
                "validation_prompts_file": "validation_prompts/text2image_prompts.txt",
                "shuffle_buffer_size": 1000,
                "num_workers": 32,
                "resolution": 256,
                "pin_memory": True,
                "persistent_workers": True
            },
            "preprocessing": {
                "max_seq_length": 512,
                "resolution": 256,
                "center_crop": False,
                "random_flip": False
            }
        },
        "optimizer": {
            "name": "adamw",
            "params": {
                "learning_rate": 5e-05,
                "scale_lr": False,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.01,
                "epsilon": 1e-8
            }
        },
        "lr_scheduler": {
            "scheduler": "cosine",
            "params": {
                "learning_rate": "${optimizer.params.learning_rate}",
                "warmup_steps": 1000
            }
        },
        "training": {
            "gradient_accumulation_steps": 1,
            "noise_type": "mask",
            "batch_size_mmu": 4,
            "mixed_precision": "bf16",
            "enable_tf32": True,
            "seed": 2025,
            "max_train_steps": 14000,
            "overfit_one_batch": False,
            "cond_dropout_prob": 0.1,
            "min_masking_rate": 0.0,
            "label_smoothing": 0.0,
            "max_grad_norm": None,
            "guidance_scale": 0.0,
            "generation_timesteps": 12,
            "mmu_coeff": 1.0
        },
        "fed": {
            "fed_alg": fed_alg,
            "test_local": False,
            "bits": 16,
            "image_aspect_ratio": "pad",
            "group_by_modality_length": True,
            "num_train_epochs": 1,
            "num_rounds": 50,
            "report_to": "tensorboard",
            
            "prox_mu": 0.01,
            "fedopt_tau": 1e-3,
            "fedopt_eta": 1e-3,
            "fedopt_beta1": 0.9,
            "fedopt_beta2": 0.99
        }
    }
    import os
    import yaml
    if not os.path.exists(os.path.dirname(yaml_path)):
        os.makedirs(os.path.dirname(yaml_path))

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
  


def cent_yaml():

    generate_cent_yaml(
        data_path=f"./data/Fed-FGVC/",
        output_dir=f"./output/Fed-FGVC/encoder_free/cent/",
        yaml_path=f"./FGVC/config_encoder_free/cent/config.yaml"

    )
  


def fed_yaml(method):
    for data_split in ["iid", "non_iid"]:
        generate_fed_yaml(
            fed_alg=method,
            data_name="Fed-FGVC",
            split_type=data_split,
            image_path=f"./data/Fed-FGVC/image/",
            data_path= f"./data/Fed-FGVC/clients/{data_split}",
            output_dir=f"./output/Fed-FGVC/encoder_free/fed/{data_split}/{method}/",
            yaml_path=f"./FGVC/config_encoder_free/fed/{data_split}/{method}/config.yaml"

        )
           



fed_yaml("fedavg")
fed_yaml("fedprox")
fed_yaml("fedyogi")
fed_yaml("fedadam")
fed_yaml("fedavgm")
fed_yaml("fedadagrad")

