from trainer.alg.eefl import EEClient, EEServer

def add_args(parser):
    parser.add_argument('--ft_step', type=int, default=5, help="Fine-tuning step")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.ft_step = args.ft_step
        self.local_keys = ['ee_fc']
        self.p_params = [any(key in name for key in self.local_keys)
                         for name, _ in self.model.named_parameters()]

    def run(self):
        self.train()

    def train(self):
        # NOTE: freeze head, update base
        for idx, param in enumerate(self.model.parameters()):
            param.requires_grad = not self.p_params[idx]
        super().train()

    def fine_tune(self):
        # NOTE: update all params
        for param in self.model.parameters():
            param.requires_grad = True

        for _ in range(self.ft_step):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)

                MODE = self.args.mode
                loss = sum(self.loss_func(preds[i], y) / self.ee_num for i in range(self.ee_num)) if MODE == 'jt' \
                    else self.loss_func(preds[self.ee_num - 1], y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def local_test(self):
        # fine-tune in the last round
        if self.server.round >= self.server.total_round - 1:
            self.fine_tune()
        super().local_test()


# extend Client to get the self.p_params
class Server(EEServer, Client):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()