import numpy as np
import torch
import cvxpy as cp

from trainer.alg.eefl import EEClient, EEServer

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=0.8, help="Alpha in weight optimization")
    parser.add_argument('--lam', type=float, default=0.01, help="Lambda in local training")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.dW = torch.zeros_like(self.model2tensor())
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
                loss += self.lam * torch.dot(w_cur, w_last) / torch.linalg.norm(w_cur)

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
        self.sims = np.zeros([self.client_num, self.client_num])
        self.graph_w = torch.zeros(self.client_num, self.client_num)
        self.alpha = args.alpha

    # NOTE: to save memory here, we will not implement a client params cache at server side, and we neglect the downlink and uplink here
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.update_sims()
        self.update_w()
        self.aggregate()

    def update_sims(self):
        sampled_ids = [client.id for client in self.sampled_clients]
        for i in sampled_ids:
            for j in sampled_ids:
                self.sims[i, j] = -torch.nn.functional.cosine_similarity(self.clients[i].dW,
                                                                         self.clients[j].dW,
                                                                         dim=0)

    # https://github.com/MediaBrain-SJTU/pFedGraph/blob/main/pfedgraph_cosine/utils.py
    def update_w(self):
        sampled_ids = [client.id for client in self.sampled_clients]
        sims_sampled = self.sims[sampled_ids, :][:, sampled_ids]

        n = len(sampled_ids)
        p = np.array([c.weight for c in self.sampled_clients])
        P = self.alpha * np.identity(n)
        P = cp.atoms.affine.wraps.psd_wrap(P)
        G = - np.identity(n)
        h = np.zeros(n)
        A = np.ones((1, n))
        b = np.ones(1)
        for idx in range(n):
            d =  sims_sampled[idx]
            q = d - 2 * self.alpha * p
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                              [G @ x <= h,
                               A @ x == b])
            prob.solve()
            self.graph_w[sampled_ids[idx], sampled_ids] = torch.Tensor(x.value)

    def aggregate(self):
        sampled_ids = [client.id for client in self.sampled_clients]

        # normalize the sampled tensors
        for r_tensor in self.received_params:
            r_tensor /= torch.linalg.norm(r_tensor)

        # aggregate
        for idx, client in enumerate(self.sampled_clients):
            w_aggr = self.graph_w[sampled_ids[idx], sampled_ids]
            aggr_tensor = sum([w * tensor for w, tensor in zip(w_aggr, self.received_params)])
            client.tensor2model(aggr_tensor)