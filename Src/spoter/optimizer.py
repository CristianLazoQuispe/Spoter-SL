class dynamic_weight_decay:
    # segunda version puede ser con la razon de cambio de la diferencia
    def __init__(self,weight_decay_patience = 15,
        weight_decay_max = 0.005,
        weight_decay_min = 0.00005):

        self.weight_decay_patience = weight_decay_patience
        self.weight_decay_max = weight_decay_max
        self.weight_decay_min = weight_decay_min

        self.patience_counter_max = 0
        self.patience_counter_min = 0

        self.loss_diff_back = 0

    def step(self,train_loss, val_loss, weight_decay):

        loss_diff = val_loss-train_loss

        if (loss_diff>self.loss_diff_back):
            # Aumentar weight decay : mas regularizacion
            self.patience_counter_max+=1
            self.patience_counter_min = 0
            if self.patience_counter_max>=self.weight_decay_patience:
                self.patience_counter_max = 0
                return min(1.5*weight_decay,self.weight_decay_max)
        if (loss_diff<self.loss_diff_back):
            self.patience_counter_min+=1
            self.patience_counter_max=0
            # Disminuir weight decay
            if self.patience_counter_min>=self.weight_decay_patience:
                self.patience_counter_min = 0
                return max(0.5*weight_decay,self.weight_decay_min)

        return weight_decay
    
    def step_v1(self,train_loss, val_loss, weight_decay):

        loss_diff = val_loss-train_loss
        if loss_diff >= 0.5:
            # Aumentar weight decay : mas regularizacion
            self.patience_counter_max+=1
            self.patience_counter_min = 0
            if self.patience_counter_max>=self.weight_decay_patience:
                self.patience_counter_max = 0
                return min(2*weight_decay,self.weight_decay_max)
        if loss_diff <= 0.45:
            self.patience_counter_min+=1
            self.patience_counter_max=0
            # Disminuir weight decay
            if self.patience_counter_min>=self.weight_decay_patience:
                self.patience_counter_min = 0
                return max(0.5*weight_decay,self.weight_decay_min)
        return weight_decay