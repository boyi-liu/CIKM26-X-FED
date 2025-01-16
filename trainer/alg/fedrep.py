from trainer.alg.eefl import EEClient, EEServer

def add_args(parser):
    parser.add_argument('--p_epoch', type=int, default=5, help="Personalized epoch")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_keys = ['ee_fc']
        self.p_params = [any(key in name for key in self.local_keys)
                         for name, _ in self.model.named_parameters()]
        self.p_epoch = args.p_epoch

    def run(self):
        self.train()

    def train(self):
        batch_loss = []
        p_batch_loss = []

        # NOTE: freeze base, update head
        for idx, param in enumerate(self.model.parameters()):
            param.requires_grad = self.p_params[idx]

        for epoch in range(self.p_epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                loss = sum(self.loss_func(preds[i], y) / self.ee_num for i in range(self.ee_num)) if self.args.mode == 'jt' \
                    else self.loss_func(preds[self.ee_num - 1], y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                p_batch_loss.append(loss.item())

        # NOTE: freeze head, update base
        for idx, param in enumerate(self.model.parameters()):
            param.requires_grad = not self.p_params[idx]

        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                loss = sum(self.loss_func(preds[i], y) / self.ee_num for i in range(self.ee_num)) if self.args.mode == 'jt' \
                    else self.loss_func(preds[self.ee_num - 1], y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


# extend Client to get the self.p_params
class Server(EEServer, Client):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()