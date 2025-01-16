from trainer.alg.eefl import EEClient, EEServer


class Client(EEClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_keys = ['ee_fc']
        self.p_params = [any(key in name for key in self.local_keys)
                         for name, _ in self.model.named_parameters()]

    def run(self):
        self.train()


# extend Client to get the self.p_params
class Server(EEServer, Client):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()