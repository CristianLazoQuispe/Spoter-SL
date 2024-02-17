class dynamic_weight_decay_diff:
    # segunda version puede ser con la razon de cambio de la diferencia
    def __init__(self,weight_decay_patience = 15,
        weight_decay_max = 0.0005,
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
                weight_decay = min((12/10.0)*weight_decay,self.weight_decay_max)
        if (loss_diff<self.loss_diff_back):
            self.patience_counter_min+=1
            self.patience_counter_max=0
            # Disminuir weight decay
            if self.patience_counter_min>=self.weight_decay_patience:
                self.patience_counter_min = 0
                weight_decay = max((10/12.0)*weight_decay,self.weight_decay_min)
        self.loss_diff_back = loss_diff
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


class dynamic_weight_decay:
    # segunda version puede ser con la razon de cambio de la diferencia
    def __init__(self,weight_decay_patience = 15,
        weight_decay_max = 0.0005,
        weight_decay_min = 0.00005,
        kp=0.0001, ki=0.0, kd=0.0,setpoint=0.5):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.setpoint = setpoint

        self.last_error = 0
        self.integral = 0

        self.weight_decay_patience = weight_decay_patience
        self.weight_decay_max = weight_decay_max
        self.weight_decay_min = weight_decay_min

    def step(self,train_loss, val_loss, weight_decay):
        current_value = val_loss-train_loss

        error = self.setpoint - current_value
        
        P = self.kp * error 
        self.integral += error
        I = self.ki * self.integral
        D = self.kd * (error - self.last_error)
        
        # Salida del PID 
        output = P + I + D
        
        # Ajustamos el weight_decay
        weight_decay += output  
        weight_decay = min(max(self.weight_decay_min, weight_decay),self.weight_decay_max)
        
        self.last_error = error
        
        return weight_decay