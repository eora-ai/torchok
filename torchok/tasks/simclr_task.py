from omegaconf import DictConfig

from torchok.constructor import TASKS
from torchok.tasks.classification import ClassificationTask


@TASKS.register_class
class SimCLR(ClassificationTask):
    """
    Task-agnostic part of the SimCLR v2 approach described in paper
    `Big Self-Supervised Models are Strong Semi-Supervised Learners`_

    .. _Big Self-Supervised Models are Strong Semi-Supervised Learners:
            https://arxiv.org/abs/2006.10029
    """
    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            pooling_name: str,
            head_name: str,
            neck_name: str = None,
            backbone_params: dict = None,
            neck_params: dict = None,
            pooling_params: dict = None,
            head_params: dict = None,
            inputs: dict = None
    ):
        super().__init__(hparams, backbone_name, pooling_name, head_name, neck_name,
                         backbone_params, neck_params, pooling_params, head_params, inputs)

    def forward_with_gt(self, batch):
        x1, x2 = batch['image_0'], batch['image_1']
        output = dict()
        output['emb1'] = self.forward(x1)
        output['emb2'] = self.forward(x2)

        return output
