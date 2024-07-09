import torch
import numpy as np
class ScheduledOptim(torch.optim.Optimizer):
    def __init__(self, model, train_config, model_config, current_step):
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=float(train_config["optimizer"]["eps"]),
            weight_decay=float(train_config["optimizer"]["weight_decay"]),
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        #self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        # self.init_lr = float(train_config["optimizer"]["init_lr"])
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        # Optimizer의 param_groups를 직접 사용
        super().__init__(self._optimizer.param_groups, {})

    def step(self, closure=None):
        self._update_learning_rate()
        return self._optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def _get_lr_scale(self):
        lr = np.min([
            np.power(self.current_step, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.current_step,
        ])
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def step_and_update_lr(self):
        self.step()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self._optimizer.state_dict()