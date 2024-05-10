from torchmetrics import Metric
import torch
import src.config as cfg

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.add_state('total_tp', default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state('total_fp', default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state('total_fn', default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=-1)

        # TP, FP, FN 계산
        for cls in range(self.num_classes):
            cls_preds = (preds == cls)
            cls_targets = (target == cls)
            tp = torch.sum(cls_preds & cls_targets).float()
            fp = torch.sum(cls_preds & ~cls_targets).float()
            fn = torch.sum(~cls_preds & cls_targets).float()
            self.total_tp[cls] += tp
            self.total_fp[cls] += fp
            self.total_fn[cls] += fn

    def compute(self):
        float_min = 1e-6        # 0으로 나누지 않기 위해 아주 작은 값을 넣어줌
        precision = self.total_tp / (self.total_tp + self.total_fp + float_min)
        recall = self.total_tp / (self.total_tp + self.total_fn + float_min)
        f1_score = 2 * (precision * recall) / (precision + recall + float_min)

        # 0이 아닌 F1 스코어만을 골라서 평균을 계산: 0인 클래스들 때문에 validation f1 score가 작게 나오는 문제 존재 --> 해결 완료
        non_zero_f1 = f1_score[f1_score > 0]
        if len(non_zero_f1) > 0:
            mean_f1_score = non_zero_f1.mean()
        else:
            mean_f1_score = torch.tensor(0.0)  # 모든 F1 스코어가 0인 경우

        return mean_f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim = -1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            ValueError("preds.shape() != target.shape()")
            return torch.tensor(0.)

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
