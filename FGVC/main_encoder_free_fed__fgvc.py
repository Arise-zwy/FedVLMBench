import torch
import copy
import os
from tqdm import tqdm
import numpy as np
import logging
import warnings
from datetime import datetime
import sys
import time
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(3000)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import *
from federated_learning import *
from config import get_config, get_training_config
from showomodel.vlm import init_vlm_model,make_supervised_data_module_clients,set_selected_state_dict
from omegaconf import OmegaConf
from pathlib import Path

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(yaml_conf, cli_conf)

    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
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
        import random
        if random.random()<0.05:
            print("label:"+label_sentence+"==> pred:"+pred_sentence)
        label_is_pos=label_sentence.find("is")
        pred_is_pos=pred_sentence.find("is")
        if label_is_pos!=-1 and pred_is_pos!=-1:
            la=label_sentence[label_is_pos+3:-14]
            answer=pred_sentence[pred_is_pos+3:-14]
            if la==answer and la!=" ":
                right+=1
        all_num+=1
    print(right,all_num)
    accuracy=right/all_num

    return {
        "accuracy": accuracy,
    }

                                                                 
config = get_config()

output_dir = config.experiment.output_dir
setup_logging(output_dir)
save_config(config)
set_seed(config.training.seed)

writer = SummaryWriter(log_dir=output_dir+'/vis')

logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
logging.info("===============================================================")
logging.info("===============================================================")
logging.info("===============================================================")


logging.info("test_local: %s",config.fed.test_local)
logging.info("batch_size: %d",config.training.batch_size_mmu) 


try:
    ## ===== laod model =====
    model,tokenizer=init_vlm_model(config)

    # ===== Load the dataset =====

    if config.dataset.dataset_name=="FGVC":
        if "variant" in config.dataset.data_path:
            config.fed.num_clients=10
            config.fed.sample_clients=10
            print("variant")
        elif "family" in config.dataset.data_path:
            config.fed.num_clients=10
            config.fed.sample_clients=10
            print("family")
        elif "manufacturer" in config.dataset.data_path:
            config.fed.num_clients=10
            config.fed.sample_clients=10
            print("manufacturer")
        else:
            ValueError("dataset error")
    if config.dataset.dataset_name=="FGVC":
        if "5clients" in config.dataset.data_path:
            config.fed.num_clients=5
            config.fed.sample_clients=5
            print("variant")
        else:
            ValueError("dataset error")


    if "non" in config.dataset.split_strategy:
        local_datasets, global_test_dataset,sub_test_datasets,data_collator = make_supervised_data_module_clients(tokenizer, config) 
    else:
        local_datasets, global_test_dataset,sub_test_datasets,data_collator = make_supervised_data_module_clients(tokenizer, config) 


    num_clients=config.fed.num_clients
    output_dir = config.experiment.output_dir

    sample_num_list = [len(local_datasets[i]) for i in range(num_clients)]
    logging.info("sample_num_list: %s",sample_num_list) 


    # ===== Define the global and local models =====

    global_dict = copy.deepcopy(model.model.state_dict())
    local_dict_list = [copy.deepcopy(global_dict) for _ in range(num_clients)]
    
    # if config.fed.fed_alg in ['fedprox','scaffold','fedadagrad', 'fedyogi', 'fedadam',"fedavgm"]:
    #     for key in global_dict.keys():
    #         global_dict[key] = global_dict[key].to(device)

    proxy_dict, opt_proxy_dict = get_proxy_dict(config.fed, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(config.fed, global_dict)


    # ===== Start federated training =====
    training_loss = [[] for _ in range(num_clients)]
    test_metrics = [{} for _ in range(num_clients)]

    select_keys=["eval_accuracy","eval_loss"]

    for round in tqdm(range(config.fed.num_rounds)):
        t_start=time.time()
        clients_this_round = get_clients_this_round(config.fed, round)

        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
        for client in range(num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue
            logging.info(f">> ============: {client} ==============")
            set_selected_state_dict(model, global_dict)  
 
            new_lr = cosine_learning_rate(round, config.fed.num_rounds, config.optimizer.params.learning_rate, 1e-6)      # manually schedule the learning rate
            training_args = get_training_config(config, new_lr)
            logging.info(f"round: {round}, new_lr: {new_lr}")

            # ===== Train local model on the client side =====
            trainer = get_fed_showo_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=local_datasets[client],
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                global_dict=global_dict,
                config=config,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                eval_dataset=None,
            )
        
            results = trainer.train()
            logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")
            training_loss[client].append(results.training_loss)
            logging.info(results)

            # ===== Client transmits local information to server =====
            if config.fed.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            # local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
            upload_params = model.model.state_dict()
            local_dict_list[client] = copy.deepcopy(upload_params)
            for key in local_dict_list[client]:
                local_dict_list[client][key] = local_dict_list[client][key].cpu()


            if config.fed.test_local and round % 3 == 0 and round >=0 :
                print("local eval")
                for idx,subdata in enumerate(sub_test_datasets):
                    eval_results = trainer.evaluate(eval_dataset=subdata)
                    with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
                        f.write(f"    ---------------- Clinet{client} : subdataset{idx}, {eval_results}\n")
                eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
                with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
                    f.write(f"     - Round {round+1} Clinet{client} : {eval_results}\n")
                    f.write("================================\n")
                                
                for key,value in eval_results.items():
                    if key in select_keys:
                        test_metrics[client][key] = eval_results[key]

            # torch.cuda.empty_cache()  
            # del trainer
            # del upload_params
            # logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

            # torch.cuda.empty_cache()
            # import gc
            # gc.collect()
            # logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")
    
        # ===== summary loss =====
        avg_loss = np.mean([training_loss[i][-1] for i in range(num_clients) if i in clients_this_round])
        writer.add_scalar('Train/Average Training Loss', avg_loss, round + 1)
        if config.fed.test_local and round % 3 == 0 and round > 5:
            for key in select_keys:
                key_values = [test_metrics[client][key] for client in range(num_clients) if key in test_metrics[client]]
                if key_values:
                    avg_key, std_key = compute_avg_and_std(key_values)
                    logging.info(f"Round {round+1}, Key: {key}, Avg: {avg_key}, Std: {std_key}")
                    writer.add_scalar(f"local_test_{key}/Avg", avg_key, round + 1)
                    writer.add_scalar(f"local_test_{key}/Std", std_key, round + 1)
                    with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
                        f.write(f"    summary - Round {round+1} : {key}, Avg: {avg_key}, Std: {std_key}\n")
  
        # ===== Server aggregates the local models =====
        use_cpu=True
        if num_clients>5:
            use_cpu=True
        global_dict, global_auxiliary = global_aggregate_showo(
            config.fed, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),use_cpu=use_cpu
        )
        set_selected_state_dict(model, global_dict)  
        torch.cuda.empty_cache()  

        # ===== Test the model =====
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 

        eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
        logging.info(eval_results)
        for key in select_keys:
            writer.add_scalar(f"Global_test/Average {key}", eval_results[key], round + 1)
        with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{current_time} - Round {round+1}: {eval_results}\n")
        if "non" in config.dataset.split_strategy:
            for idx,subdata in enumerate(sub_test_datasets):
                eval_results = trainer.evaluate(eval_dataset=subdata)
                with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
                    f.write(f"    ---------------- subdataset{idx}, {eval_results}\n")
            with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
                f.write("================================\n")
                f.write("================================\n")

        # if (round+1) % config.experiment.save_model_freq == 0:
        #     logging.info(f"checkpoint-{round+1}")
            # trainer.save_model(os.path.join(output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(output_dir, "training_loss.npy"), np.array(training_loss))
        t_end=time.time()
        logging.info(f"round time: {(t_end-t_start)/3600:.2f} hours")

    with open("running_med.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, \n")
        f.write("======================================================\n")


except Exception as e:
    logging.error(e)
    logging.error("error")
    with open("running_med.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, \n")
        f.write("======================================================\n")
    raise e