
class WandBID:
    def __init__(self, wandb_id):
        self.wandb_id = wandb_id

    def state_dict(self):
        return self.wandb_id

class Epoch:
    def __init__(self, epoch):
        self.epoch = epoch

    def state_dict(self):
        return self.epoch
    
class Metric:
    def __init__(self, metric):
        self.metric = metric

    def state_dict(self):
        return self.metric