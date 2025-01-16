import torch

from models.config import load_model
from trainer.alg.eefl import EEClient, EEServer
from utils.loss_utils import balanced_softmax_loss


def add_args(parser):
    parser.add_argument('--p_epoch', type=int, default=5, help="Personalized epoch")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.p_epoch = args.p_epoch
        self.local_keys = ['ee_fc']
        self.head_params = [any(key in name for key in self.local_keys)
                            for name, _ in self.model.named_parameters()]

        self.p_model = load_model(args).to(args.device)
        self.p_optim = torch.optim.SGD(params=self.p_model.parameters(),
                                       lr=self.lr,
                                       momentum=0.9,
                                       weight_decay=1e-4)


        self.g_head = self.model2tensor(params=self.head_params)
        self.p_head = torch.zeros_like(self.g_head)

        self.sample_per_class = torch.zeros(args.class_num)
        for x, y in self.loader_train:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

    def run(self):
        self.train()
        self.copy_backbone()
        self.p_train()

    def train(self):
        batch_loss = []

        # ===== update global params =====
        # NOTE: activate all params
        for idx, param in enumerate(self.model.parameters()):
            param.requires_grad = True

        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)
                MODE = self.args.mode
                loss = sum(balanced_softmax_loss(labels=y,
                                                 logits=preds[i],
                                                 sample_per_class=self.sample_per_class) / self.ee_num for i in range(self.ee_num)) if MODE == 'jt' \
                    else balanced_softmax_loss(labels=y,
                                               logits=preds[self.ee_num - 1],
                                               sample_per_class=self.sample_per_class)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def p_train(self):
        p_batch_loss = []

        # NOTE: freeze base
        for idx, param in enumerate(self.p_model.parameters()):
            param.requires_grad = self.head_params[idx]

        for epoch in range(self.p_epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)

                preds_g = self.model(X)
                preds_p = self.p_model(X)

                MODE = self.args.mode
                loss = sum(self.loss_func(preds_p[i] + preds_g[i].detach(), y) / self.ee_num for i in range(self.ee_num)) if MODE == 'jt' \
                    else self.loss_func(preds_p[self.ee_num-1] + preds_g[self.ee_num-1].detach(), y)

                self.p_optim.zero_grad()
                loss.backward()
                self.p_optim.step()

                p_batch_loss.append(loss.item())

    def local_test(self):
        self.model.load_state_dict(self.p_model.state_dict())
        return super().local_test()

    def copy_backbone(self):
        for g_param, p_param, is_head in zip(self.model.parameters(), self.p_model.parameters(), self.head_params):
            if is_head: continue
            p_param.data = g_param.data.clone()

class Server(EEServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()