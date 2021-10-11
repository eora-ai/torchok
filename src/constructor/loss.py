import torch.nn as nn

from src.constructor.config_structure import LossParams
from src.registry import LOSSES


class Wrapper:
    def __init__(self, obj):
        self.__obj = obj

    @property
    def training(self):
        return self.__obj.training

    def log(self, *args, **kwargs):
        self.__obj.log(*args, **kwargs)


class JointLoss(nn.Module):
    def __init__(self, task_module, params: LossParams):
        super(JointLoss, self).__init__()
        self.task_module = Wrapper(task_module)
        self.log_separate_losses = params.log_separate_losses

        loss_list = params.loss_list
        loss_weights = params.weights
        self.losses = nn.ModuleList()

        for criterion_params in loss_list:
            criterion_class = LOSSES.get(criterion_params.name)
            target_fields = criterion_params.params.pop('target_fields', None)
            if target_fields is None:
                raise AttributeError('`target_fields` must be specified in loss parameters')
            name = criterion_params.params.pop('name', criterion_params.name)
            criterion = criterion_class(**criterion_params.params)
            criterion.target_fields = target_fields
            criterion.name = name
            self.losses.append(criterion)

        if loss_weights is None:
            self.weights = [1] * len(self.losses)
        else:
            if len(self.losses) != len(loss_weights):
                raise ValueError('Length of weights must be equal to the number of losses or be None')
            weights_sum = sum(loss_weights) / len(self.losses)
            self.weights = [w / weights_sum for w in loss_weights]

    def forward(self, **kwargs):
        total_loss = 0
        active_loss = False
        for i, loss_module in enumerate(self.losses):
            targeted_kwargs = self.map_arguments(loss_module.target_fields, kwargs)
            if targeted_kwargs:
                loss = loss_module(**targeted_kwargs)
                if self.log_separate_losses:
                    mode = 'train' if self.task_module.training else 'valid'
                    self.task_module.log(f'{mode}/{loss_module.name}', loss,
                                         on_step=False, on_epoch=True)
                active_loss = True
            else:
                loss = 0
            total_loss = total_loss + loss * self.weights[i]
        if not active_loss:
            raise RuntimeError('Expected arguments to pass in at least one loss in loss list')
        return total_loss

    @staticmethod
    def map_arguments(target_fields, kwargs):
        targeted_kwargs = {}
        for target_arg, source_arg in target_fields.items():
            if source_arg in kwargs:
                targeted_kwargs[target_arg] = kwargs[source_arg]
        return targeted_kwargs
