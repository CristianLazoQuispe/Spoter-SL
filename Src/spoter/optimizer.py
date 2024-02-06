class dynamic_weight_decay:

    def __init__(self,weight_decay_patience = 15,
        weight_decay_max = 0.001,
        weight_decay_min = 0.000001):

        self.weight_decay_patience = weight_decay_patience
        self.weight_decay_max = weight_decay_max
        self.weight_decay_min = weight_decay_min

        self.patience_counter_max = 0
        self.patience_counter_min = 0

    def step(self,train_loss, val_loss, weight_decay):

        loss_diff = val_loss-train_loss
        if loss_diff >= 0.7:
            # Aumentar weight decay : mas regularizacion
            self.patience_counter_max+=1
            self.patience_counter_min = 0
            if self.patience_counter_max>=self.weight_decay_patience:
                self.patience_counter_max = 0
                return min(1.05*weight_decay,self.weight_decay_max)
        if loss_diff <= 0.3:
            self.patience_counter_min+=1
            self.patience_counter_max=0
            # Disminuir weight decay
            if self.patience_counter_min>=self.weight_decay_patience:
                self.patience_counter_min = 0
                return max(0.95*weight_decay,self.weight_decay_min)
        return weight_decay