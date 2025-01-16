import numpy as np
import torch
import cvxpy as cp

from trainer.alg.eefl import EEClient, EEServer
from utils.loss_utils import kd_loss_func

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=0.6, help="Alpha")
    parser.add_argument('--lam', type=float, default=1, help="Alpha")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_keys = ['ee_fc']
        self.p_params = [any(key in name for key in self.local_keys)
                         for name, _ in self.model.named_parameters()]
        self.cur_scale = 0
        self.ee_num = args.ee_num
        self.lam = args.lam
        self.loss_list = [0 for _ in range(self.ee_num)]
        self.weights_kd = [0 for _ in range(self.ee_num)]
        self.dW = self.classifier2tensor(self.ee_num-1)

    def run(self):
        self.train()

    def train(self):
        self.last_classifier_tensor = self.classifier2tensor(self.ee_num-1)

        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                loss = 0
                for i in range(self.cur_scale):
                    loss_i = self.loss_func(preds[i], y)
                    self.loss_list[i] = loss_i.item()
                    loss += loss_i / self.ee_num
                loss += self.loss_func(preds[self.ee_num - 1], y) / self.ee_num
                kd_loss = 0

                for i, student in enumerate(preds):
                    if i > self.cur_scale: continue
                    else:
                        kd_loss += kd_loss_func(student, preds[self.ee_num-1].detach())
                loss += self.lam * kd_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

        self.dW = self.last_classifier_tensor - self.classifier2tensor(self.ee_num-1)


    def classifier2tensor(self, idx):
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        classifiers = self.model.ee_classifiers
        return nan_to_zero(
            torch.cat([param.data.view(-1) for param in classifiers[idx].parameters()], dim=0)
        )

    def tensor2classifier(self, tensor, idx):
        param_index = 0
        for param in self.model.ee_classifiers[idx].parameters():
            param_size = param.numel()
            param.data = tensor[param_index: param_index + param_size].view(param.shape).detach().clone()
            param_index += param_size


# extend Client to get the self.p_params
class Server(EEServer, Client):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.sims = np.zeros([self.client_num, self.client_num])
        self.graph_w = torch.zeros(self.client_num, self.client_num)
        self.alpha = args.alpha
        self.exit_ratio = 0

    def run(self):
        self.sample()
        self.schedule()
        self.downlink()
        self.client_update()
        self.uplink()
        self.update_sims()
        self.update_graph_w()
        self.aggregate()

    def schedule(self):
        sampled_ids = [client.id for client in self.sampled_clients]
        self.conflict_matrix = self.sims[sampled_ids, :][:, sampled_ids]
        self.conflict_matrix[self.conflict_matrix > 0] = 0

        self.set_exit_ratio()

        client_scale = len(self.sampled_clients)
        total_exit_num = self.ee_num * client_scale
        to_select_exit_num = int(total_exit_num * self.exit_ratio)
        selected_exit_num = 0
        print(f'To select {to_select_exit_num} exits at round {self.round}')
        for ee_idx in range(self.ee_num):
            to_select = min(to_select_exit_num - selected_exit_num, client_scale)
            if to_select == 0:
                break
            if to_select == client_scale:
                for c in self.sampled_clients:
                    c.cur_scale = ee_idx
            else:
                conflict_degree_per_client = np.sum(self.conflict_matrix, axis=0)
                selected_clients = np.argpartition(conflict_degree_per_client, -to_select)[-to_select:]
                for idx in selected_clients:
                    self.sampled_clients[idx].cur_scale = ee_idx
            selected_exit_num += to_select


    def update_sims(self):
        for client_i in self.sampled_clients:
            for client_j in self.sampled_clients:
                self.sims[client_i.id, client_j.id] = torch.nn.functional.cosine_similarity(
                    client_i.dW,
                    client_j.dW,
                    dim=0)

    def update_graph_w(self):
        sampled_ids = [client.id for client in self.sampled_clients]
        sims_sampled = self.sims[sampled_ids, :][:, sampled_ids]

        n = len(sampled_ids)
        P = np.identity(n)
        P = cp.atoms.affine.wraps.psd_wrap(P)
        G = - np.identity(n)
        h = np.zeros(n)
        A = np.ones((1, n))
        b = np.ones(1)
        for idx in range(n):
            d = sims_sampled[idx] / self.alpha
            p = np.array([1 / n for _ in self.sampled_clients])
            q = -d - 2 * p

            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),[G @ x <= h, A @ x == b])
            prob.solve()
            self.graph_w[sampled_ids[idx], sampled_ids] = torch.Tensor(x.value)

    def set_exit_ratio(self):
        sampled_ids = [client.id for client in self.sampled_clients]
        conflict_degree = np.sum(self.conflict_matrix)

        exit_ratio = 1 - 1/self.ee_num
        basic_ratio = 0
        end_round = self.args.rnd / 2
        self.exit_ratio = basic_ratio + self.round / end_round * (exit_ratio - basic_ratio) + conflict_degree / (len(sampled_ids) * len(sampled_ids))
        self.exit_ratio = min(exit_ratio, self.exit_ratio)
        self.exit_ratio = max(basic_ratio, self.exit_ratio)

    def aggregate(self):
        super().aggregate()
        sampled_ids = [client.id for client in self.sampled_clients]
        scale = self.ee_num - 1
        for client in self.sampled_clients:
            w_aggr = self.graph_w[client.id, sampled_ids]
            aggr_tensor_list = [client.classifier2tensor(scale) for client in self.sampled_clients]

            print(f'weight of client-{client.id}: {w_aggr}')
            aggr_tensor = sum([w * tensor for w, tensor in zip(w_aggr, aggr_tensor_list)])
            client.tensor2classifier(aggr_tensor, scale)