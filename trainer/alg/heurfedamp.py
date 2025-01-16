import math

import numpy as np
import torch

from trainer.alg.eefl import EEClient, EEServer
from trainer.base import BaseServer

def add_args(parser):
    parser.add_argument('--lam', type=float, default=1, help="Lambda in local training")
    parser.add_argument('--sigma', type=float, default=10, help="Sigma in server aggregation")
    parser.add_argument('--xiii', type=float, default=0.5, help="xi_{ii}")
    return parser.parse_args()


class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.lam = args.lam

    def run(self):
        self.train()

    # model is directly set in aggregate() of server for saving memory
    # no need to implement a clone_model()
    def clone_model(self, target):
        pass

    def train(self):
        w_last = self.model2tensor()

        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                MODE = self.args.mode
                loss = sum(self.loss_func(preds[i], y) / self.ee_num for i in range(self.ee_num)) if MODE == 'jt' \
                    else self.loss_func(preds[self.ee_num - 1], y)

                w_cur = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)
                loss += 0.5 * self.lam * torch.norm(w_cur - w_last, p=2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))
        self.dW = self.model2tensor() - w_last


class Server(EEServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.sims = torch.zeros(self.client_num, self.client_num)
        self.sigma = args.sigma
        self.xiii = args.xiii

    # NOTE: to save memory here, we will not implement a client params cache at server side, and we neglect the downlink and uplink here
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.update_sims()
        self.aggregate()

    def update_sims(self):
        sampled_ids = [client.id for client in self.sampled_clients]
        for i in sampled_ids:
            for j in sampled_ids:
                self.sims[i, j] = torch.nn.functional.cosine_similarity(self.clients[i].dW,
                                                                        self.clients[j].dW,
                                                                        dim=0)

    def aggregate(self):
        sampled_ids = [client.id for client in self.sampled_clients]
        for idx, client in enumerate(self.sampled_clients):
            w_aggr = self.sims[sampled_ids[idx], sampled_ids]

            w_aggr = torch.exp(w_aggr * self.sigma)
            w_aggr /= (torch.sum(w_aggr)-w_aggr[idx])
            w_aggr *= (1-self.xiii)
            w_aggr[idx] = self.xiii
            aggr_tensor = sum([w * tensor for w, tensor in zip(w_aggr, self.received_params)])
            client.tensor2model(aggr_tensor)