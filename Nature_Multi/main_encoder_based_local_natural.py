import torch
import os
from tqdm import tqdm
import numpy as np
import logging
import warnings
import datetime
import sys
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(2000)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module_mix_local
import warnings
from pycocoevalcap import compute_scores
import random
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_acc_metrics(eval_pred):
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

    if all_labels[0].find("caption")!=-1:
        gts={}
        res={}
        iid=0
    
        for label_sentence, pred_sentence in zip(all_labels, all_preds):          
            label_is_pos = label_sentence.find("is")
            pred_is_pos = pred_sentence.find("is")
            if label_is_pos != -1 and pred_is_pos != -1:
                gts[iid]=[label_sentence[label_is_pos + 4:-5]]
                res[iid]=[pred_sentence[pred_is_pos + 4:-5]]
                iid+=1
                if iid==10:
                    print(gts,res)
        result=compute_scores(gts, res, method="English_word", tokenizer=None, temp=True)
        print("len:",len(gts))
        return {
            "CIDEr": result["CIDEr"],
            "ROUGE_L": result["ROUGE_L"],
        }    


    elif all_labels[0].find("classification")!=-1:
        right=0
        all_num=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random()<0.005:
                print("label:",label_sentence)
                print("pred:",pred_sentence)
            label_is_pos=label_sentence.find("is")
            pred_is_pos=pred_sentence.find("is")
            if label_is_pos!=-1 and pred_is_pos!=-1:
                la=label_sentence[label_is_pos+3:label_is_pos+4]
                answer=pred_sentence[pred_is_pos+3:pred_is_pos+4]
                if la==answer and la!=" ":
                    right+=1
            all_num+=1

        accuracy=right/all_num
        print(right,all_num,accuracy)

        return {
            "class_accuracy": accuracy,
        }
    elif all_labels[0].find("question")!=-1:
        right=0
        all_num=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random()<0.005:
                print("label:",label_sentence)
                print("pred:",pred_sentence)
            label_is_pos=label_sentence.find("is")
            pred_is_pos=pred_sentence.find("is")
            if label_is_pos!=-1 and pred_is_pos!=-1:
                la=label_sentence[label_is_pos+3:label_is_pos+4]
                answer=pred_sentence[pred_is_pos+3:pred_is_pos+4]
                if la==answer and la!=" ":
                    right+=1
            all_num+=1

        accuracy=right/all_num
        print(right,all_num,accuracy)

        return {
            "vqa_accuracy": accuracy,
        }
    elif all_labels[0].find("coordinates")!=-1:
        print(all_preds[0:5])
        iou_scores = []
        len_ok=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            label_is_pos = label_sentence.find("are")
            pred_is_pos = pred_sentence.find("are")
            if label_is_pos != -1 and pred_is_pos != -1:
                len_ok+=1
                label_pos = label_sentence[label_is_pos + 3:-5]
                answer_pos = pred_sentence[pred_is_pos + 3:-5]
              
                if label_pos and answer_pos and is_valid_box_format(label_pos) and is_valid_box_format(answer_pos):
                    import ast
                    label_boxes = ast.literal_eval(label_pos)
                    try:
                        pred_boxes = ast.literal_eval(answer_pos)
                    except ValueError as e:
                        print(f"Error parsing answer_pos: {answer_pos}")
                        pred_boxes = None  

                    if pred_boxes != None and len(label_boxes) == 4 and len(pred_boxes) == 4:
                        label_boxes = [label_boxes]
                        pred_boxes = [pred_boxes]
                        for label_box, pred_box in zip(label_boxes, pred_boxes):
                            # print("label_boxes:",label_box,type(label_box))
                            # print("pred_boxes:",pred_box,type(pred_box))

                            if isinstance(label_box, dict) and isinstance(pred_box, dict):
                                for key in label_box:
                                    if key in pred_box:
                                        iou = compute_iou(label_box[key], pred_box[key])
                                        iou_scores.append(iou)
                                        if random.random() < 0.05:
                                            print("label_box:",label_box)
                                            print("pred_box:",pred_box)
                                            print("iou:",iou)
                                            print("=====================================")
                            elif isinstance(label_box, list) and isinstance(pred_box, list):
                                iou = compute_iou(label_box, pred_box)
                                iou_scores.append(iou)
                                if random.random() < 0.05:
                                    print("label_box:",label_box)
                                    print("pred_box:",pred_box)
                                    print("iou:",iou)
                                    print("=====================================")

        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        print("len:",len(iou_scores))
        print("len_ok:",len_ok)

        return {
            "mean_iou": mean_iou,
        }
    else:
        return {
                "mean_iou": 0,
            }

# ===== Define the arguments =====
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()
config_file=args.config
script_args, fed_args, peft_config, model_args, data_args = get_config(config_file)

training_args = get_training_args(script_args, script_args.learning_rate)
script_args = save_config(script_args, fed_args, model_args, data_args)
set_seed(script_args.seed)

logdir = script_args.output_dir  
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
print(logdir)
writer = SummaryWriter(log_dir=logdir+'/vis')

with open("running_nature_local.txt","a") as f:
    f.write(f"time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")

logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")


try:
    # training_args.max_steps=len(local_datasets)//training_args.batch_size
    # script_args.max_steps=len(local_datasets)//training_args.batch_size
    logging.info("max_steps: %d",training_args.max_steps) 
    logging.info("batch_size: %d",training_args.batch_size) 
    logging.info("learning rate: %d",script_args.learning_rate) 

    ## ===== laod lamma =====
    model,tokenizer=init_vlm_model(script_args,model_args, data_args)
    # ===== Load the dataset =====
    local_datasets, global_test_datasets,data_collator = make_supervised_data_module_mix_local(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) 
    logging.info("sample_num_list: %s",len(local_datasets)) 

    if len(local_datasets)>2000:
        script_args.logging_steps=50

    # ===== Get model config =====
    device_map, quantization_config, torch_dtype = get_model_config(script_args)

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # if training_args.gradient_checkpointing:
    #     model.enable_input_require_grads()
        
    resume_from_checkpoint=training_args.resume_from_checkpoint

    # ===== Define the tokenizer =====
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # ===== Start federated training =====
    training_loss = []

    for round in tqdm(range(fed_args.num_rounds)):

        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)
        logging.info(f"round: {round}, new_lr: {new_lr}")

        # ===== Train local model on the client side =====
        trainer = get_local_vlm_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=local_datasets,
            eval_dataset=None,
            compute_metrics=compute_acc_metrics,
            data_collator=data_collator,
            fed_args=fed_args,
            script_args=script_args,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            resume_from_checkpoint=resume_from_checkpoint if round == 0 else None  # Only resume from checkpoint in the first round
        )
        logging.info(f"Current memory allocated: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

        results = trainer.train()
        logging.info(results)

        training_loss.append(results.training_loss)
        writer.add_scalar('Train/Average Training Loss', results.training_loss, round + 1)

        # ===== Test the model =====
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        select_keys=["eval_class_accuracy","eval_vqa_accuracy","eval_loss", "eval_mean_iou","eval_CIDEr","eval_ROUGE_L"]

        for area_dataset in global_test_datasets:
            eval_results = trainer.evaluate(eval_dataset=area_dataset)
            logging.info(area_dataset.dataset_name ,eval_results)
            for key in select_keys:
                if key in eval_results:
                    writer.add_scalar(f"Global_test/Average {key}", eval_results[key], round + 1)
            with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if "eval_vqa_accuracy" in eval_results:
                    task_name = "VQA"
                elif "eval_class_accuracy" in eval_results:
                    task_name = "classification"
                elif "eval_mean_iou" in eval_results:
                    task_name = "Detection"
                else:
                    task_name = "Caption"
                f.write(f"{current_time} - Round {round+1}: {task_name} {eval_results}\n")

        if (round+1) % fed_args.save_model_freq == 0:
            logging.info(f"checkpoint-{round+1}")
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    with open("running_nature_local.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")


except Exception as e:
    logging.error(e)
    logging.error("error")
    with open("running_nature_local.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")
    raise e