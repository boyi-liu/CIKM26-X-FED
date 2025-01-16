from trainer.alg.eefl import EEClient, EEServer
from utils.loss_utils import kd_loss_func

def add_args(parser):
    parser.add_argument('--beta', type=float, default=0.5, help="Weight of KL loss")
    return parser.parse_args()

class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.beta = args.beta

    def run(self):
        self.train()

    def train(self):
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = 0

                teacher_logits = preds[self.ee_num - 1]
                for i in range(self.ee_num):
                    ce_loss = self.loss_func(preds[i], y)
                    kl_loss = kd_loss_func(y=preds[i],
                                           teacher_scores=teacher_logits.detach(),
                                           T=1) * self.beta
                    loss += i * (ce_loss + kl_loss)
                loss /= (self.ee_num * (self.ee_num + 1))

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