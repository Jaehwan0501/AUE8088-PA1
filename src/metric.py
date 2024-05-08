from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_precision', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_recall', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_f1', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=-1)

        # TP, FP, FN 계산
        tp = torch.sum((preds == 1) & (target == 1)).float()
        fp = torch.sum((preds == 1) & (target == 0)).float()
        fn = torch.sum((preds == 0) & (target == 1)).float()

        float_min = 1e-6

        precision = tp / (tp + fp + float_min)
        recall = tp / (tp + fn + float_min)
        f1_score = 2 * (precision * recall) / (precision + recall + float_min)

        # Accumulate to self.xxxx
        self.total_precision += precision
        self.total_recall += recall
        self.total_f1 += f1_score
        self.count += 1

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.)
        return self.total_f1.float() / self.count.float()

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
