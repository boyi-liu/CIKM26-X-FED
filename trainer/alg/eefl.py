import torch
import torch.nn.functional as F

from trainer.base import BaseClient, BaseServer
from utils.metric_utils import DataProcessor


class EEClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.ee_num = args.ee_num
        for idx in range(self.ee_num):
            self.metric[f'acc_{idx}'] = DataProcessor()

    def train(self):
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = 0

                MODE = self.args.mode
                KD = True
                def kd_loss_func(y, teacher_scores, T=2):
                    p = F.log_softmax(y/T, dim=1)
                    q = F.softmax(teacher_scores/T, dim=1)
                    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
                    return l_kl

                if MODE == 'si':
                    loss = self.loss_func(preds[self.ee_num - 1], y)
                elif MODE == 'jt':
                    loss = sum(self.loss_func(preds[i], y) / self.ee_num for i in range(self.ee_num))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def local_test(self):
        self.model.eval()
        total = 0
        correct_list = [0 for _ in range(self.ee_num)]
        with torch.no_grad():
            for data in self.loader_test:
                X, y = self.preprocess(data)
                preds = self.model(X)
                total += y.size(0)
                for i in range(self.ee_num):
                    _, predicted = torch.max(preds[i].data, 1)
                    correct_list[i] += (predicted == y).sum().item()

        acc = [100.00 * correct / total for correct in correct_list]
        for idx in range(self.ee_num):
            self.metric[f'acc_{idx}'].append(acc[idx])
        self.metric['acc'].append(sum(acc)/len(acc))

class EEServer(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)

        self.ee_num = args.ee_num
        for idx in range(self.ee_num):
            self.metric[f'acc_{idx}'] = DataProcessor()