import torch

from trainer.alg.eefl import EEClient, EEServer

def add_args(parser):
    parser.add_argument('--mu', type=float, default=0.05, help="Mu")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.mu = args.mu

    def run(self):
        self.train()

    def train(self):
        gm = self.model2tensor() # this is only param.data, without grad

        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                pm = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)
                MODE = self.args.mode
                loss = sum(self.loss_func(preds[i], y) / self.ee_num for i in range(self.ee_num)) if MODE == 'jt' \
                    else self.loss_func(preds[self.ee_num - 1], y)
                loss += self.mu * torch.norm(gm - pm, p=2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


class Server(EEServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()