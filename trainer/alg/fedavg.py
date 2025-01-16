from trainer.alg.eefl import EEClient, EEServer

class Client(EEClient):
    def run(self):
        self.train()


class Server(EEServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
