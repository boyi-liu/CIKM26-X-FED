import torch

from models.config import load_model
from trainer.alg.eefl import EEClient, EEServer

def add_args(parser):
    parser.add_argument('--p_epoch', type=int, default=5, help="Epoch for personalized part")
    parser.add_argument('--lam', type=float, default=0.1, help="Lambda")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.lam = args.lam
        self.p_epoch = args.p_epoch
        self.p_model = load_model(args).to(args.device)
        self.p_optim = torch.optim.SGD(params=self.p_model.parameters(),
                                       lr=self.lr,
                                       momentum=0.9,
                                       weight_decay=1e-4)

    def run(self):
        self.train()
        self.p_train()

    def p_train(self):
        self.p_model.train()
        gm = self.server.model2tensor()
        # === train ===
        for epoch in range(self.p_epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.p_model(X)

                pm = torch.cat([p.view(-1) for p in self.p_model.parameters()], dim=0)
                MODE = self.args.mode
                loss = sum(self.loss_func(preds[i], y) / self.ee_num for i in range(self.ee_num)) if MODE == 'jt' \
                    else self.loss_func(preds[self.ee_num - 1], y)
                loss += self.lam / 2 * torch.norm(gm - pm, p=2)

                self.p_optim.zero_grad()
                loss.backward()
                self.p_optim.step()

    def local_test(self, g_test=False):
        if not g_test:
            self.model.load_state_dict(self.p_model.state_dict())
        return super().local_test()

    def reset_optimizer(self, decay=True):
        super().reset_optimizer(decay)
        for param_group in self.p_optim.param_groups:
            param_group['lr'] = self.lr * (self.args.gamma ** self.server.round)

class Server(EEServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()