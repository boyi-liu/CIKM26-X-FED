from trainer.alg.eefl import EEClient, EEServer


class Client(EEClient):
    def run(self):
        self.train()

    def clone_model(self, target):
        # NOTE: no downlink here
        pass

# extend Client to get the self.p_params
class Server(EEServer):
    def run(self):
        self.sample()
        self.client_update()