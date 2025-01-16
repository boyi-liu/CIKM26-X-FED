import torch

from trainer.alg.eefl import EEClient, EEServer
from utils.loss_utils import kd_loss_func

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=0.1, help="Regularization for FedDyn")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.alpha = args.alpha
        self.prev_grad = torch.zeros_like(self.model2tensor())

    def run(self):
        self.train()

    def train(self):
        gm = self.model2tensor()

        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                ce_loss = 0
                kl_loss = 0
                dyn_loss = 0

                for i in range(self.ee_num):
                    ce_loss += self.loss_func(preds[i], y)
                    for j in range(self.ee_num):
                        if j == i:
                            continue
                        kl_loss += kd_loss_func(y=preds[i],
                                                teacher_scores=preds[j].detach(),
                                                T=1)
                l_tensor = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)
                dyn_loss -= torch.dot(l_tensor, self.prev_grad)
                dyn_loss += self.alpha / 2 * torch.norm(l_tensor - gm, p=2)

                loss = ce_loss + kl_loss / (self.ee_num-1) + dyn_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

        self.prev_grad -= self.alpha*(self.model2tensor() - gm)

class Server(EEServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
