import torch
import copy
import os
from tqdm import tqdm
import numpy as np
import logging
import datetime
import sys
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(2000)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import *
from federated_learning import *
from config import get_config, get_training_config

from showomodel.vlm import init_vlm_model,make_supervised_data_module_local
from omegaconf import OmegaConf
from pathlib import Path


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(yaml_conf, cli_conf)

    now_time = (datetime.datetime.now()).strftime("%Y%m%d%H%M%S")
    config.experiment.output_dir = f"{config.experiment.output_dir}/{config.dataset.dataset_name}_{config.dataset.split_strategy}_{config.fed.fed_alg}_L{config.optimizer.params.learning_rate}__b{config.training.batch_size_mmu}_l{config.dataset.preprocessing.max_seq_length}_{now_time}"

    return config

def save_config(config):
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    config_path = Path(config.experiment.output_dir) / "config.yaml"
    logging.info(f"Saving config to {config_path}")
    OmegaConf.save(config, config_path)

def setup_logging(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = os.path.join(logdir, 'training.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
    
    logging.info("logdir: %s", logfile)

def compute_metrics(eval_pred):
   
    labels = eval_pred.label_ids
    preds = eval_pred.predictions[0]

    all_labels = []
    all_preds = []

    for case_labels, case_preds in zip(labels, preds):
        case_lab,case_pred=[],[]
        for label, pred in zip(case_labels, case_preds):
            if label != -100:
                case_lab.append(label)
                case_pred.append(pred)
        all_labels.append(tokenizer.decode(case_lab))
        all_preds.append(tokenizer.decode(case_pred))


    right=0
    all_num=0
    for label_sentence, pred_sentence in zip(all_labels, all_preds):
        label_is_pos=label_sentence.find("is")
        pred_is_pos=pred_sentence.find("is")
        if label_is_pos!=-1 and pred_is_pos!=-1:
            la=label_sentence[label_is_pos+3:-5]
            answer=pred_sentence[pred_is_pos+3:-5]
            # print("'"+la+"'","'"+answer+"'")
            if la==answer and la!=" ":
                right+=1
        all_num+=1
    print(right,all_num)
    accuracy=right/all_num

    return {
        "accuracy": accuracy,
    }

# ===== Define the arguments =====
config = get_config()

output_dir = config.experiment.output_dir
setup_logging(output_dir)
save_config(config)
set_seed(config.training.seed)


logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Current memory allocated: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")


try:
        
    ## ===== laod lamma =====
    model,tokenizer=init_vlm_model(config)
    # ===== Load the dataset =====
    local_datasets, global_test_dataset,data_collator = make_supervised_data_module_local(tokenizer, config) 



    # ===== Start federated training =====
    training_loss = []

    for round in tqdm(range(config.fed.num_rounds)):

        new_lr = cosine_learning_rate(round, config.fed.num_rounds, config.optimizer.params.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_config(config, new_lr)
        logging.info(f"round: {round}, new_lr: {new_lr}")

        # ===== Train local model on the client side =====
        trainer = get_local_showo_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=local_datasets,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            config=config,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

        results = trainer.train()
        logging.info(results)

        training_loss.append(results.training_loss)

    
        # ===== Save the model =====
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 

    
        eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
        logging.info(eval_results)
        with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{current_time} - Round {round+1}: {eval_results}\n")
        

        
        np.save(os.path.join(output_dir, "training_loss.npy"), np.array(training_loss))
    with open("running.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}\n")
        f.write("======================================================\n")


except Exception as e:
    logging.error(e)
    logging.error("error")
    with open("running.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}\n")
        f.write("======================================================\n")

    raise e