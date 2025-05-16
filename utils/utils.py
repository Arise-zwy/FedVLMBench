import math
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import numpy as np
import random
def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


def calculate_matching_accuracy(labels, preds):
    # 提取最后的选项
    labels_option = labels[-1]
    preds_option = preds[-1]

    # 计算 precision、recall、f1 和 support
    precision, recall, f1, _ = precision_recall_fscore_support([labels_option], [preds_option], average='binary')

    # 计算准确率
    acc = accuracy_score([labels_option], [preds_option])

    return precision, recall, f1, acc

class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def preprocess_logits_for_metrics(logits, labels):
    predictions = torch.argmax(logits, dim=-1)

    # print("testing")
    # print(predictions.shape)
    # print(labels.shape)
    # print("predictions",predictions[0].shape,predictions[0][420:])
    # print("labels",labels[0].shape,labels[0])
    return predictions, labels

def compute_avg_and_std(values):
    values=np.array(values)
    avg=np.mean(values)
    std=np.std(values)
    return avg,std

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_iou(box1, box2):
    try:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1, y1, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2, y2, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou
    except:
        print("ERROR",box1,box2)
        print("!!!!!!!!!!!!!!!!")
        return 0.0

def is_valid_box_format(box_str):
    try:
        box_list = eval(box_str)
        if isinstance(box_list,dict) and len(box_list)==1:
            return True
        if isinstance(box_list, list) and all(isinstance(item, dict) for item in box_list):
            return True
        elif isinstance(box_list, list) and len(box_list) == 4 and all(isinstance(item, float) for item in box_list):
            return True
       
    except:
        return False
    return False



if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
