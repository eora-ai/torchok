import torch
from torch.nn import Module, CrossEntropyLoss
from torchok.constructor import LOSSES


@LOSSES.register_class
class NT_XentLoss(Module):
    """
    Unsupervised loss for task-agnostic part of the SimCLR v2 approach described in paper
    `Big Self-Supervised Models are Strong Semi-Supervised Learners`_

    Args:
        reduction: as in PyTorch's CrossEntropyLoss
        temperature: temperature for scaling logits

    .. _Big Self-Supervised Models are Strong Semi-Supervised Learners:
            https://arxiv.org/abs/2006.10029
    """

    def __init__(self, reduction: str = "mean", temperature: float = 1.0) -> None:
        super().__init__()
        self.ce = CrossEntropyLoss(weight=None, reduction=reduction)
        self.temperature = temperature

    def forward(self, emb1, emb2, emb_m=None):
        batch_size = emb1.shape[0]
        device = emb1.device

        if emb_m is None:
            emb = torch.cat([emb1, emb2])
            # Calculate similarity matrix so that it will have view of a block matrix,
            # where A - items of the first branch, B - items of the second branch
            # [[A*A, A*B]
            #  [B*A, B*B]]
            sim_mat = torch.matmul(emb, emb.transpose(1, 0))
        else:
            emb_left = torch.cat([emb1, emb2])
            emb_right = torch.cat([emb1, emb2, emb_m])
            # Calculate similarity matrix so that it will have view of a block matrix,
            # where A - items of the first branch, B - items of the second branch
            # [[A*A, A*B, A*M]
            #  [B*A, B*B, B*M]]
            sim_mat = torch.matmul(emb_left, emb_right.transpose(1, 0))

        sim_mat /= self.temperature
        sim_mat[: batch_size * 2, : batch_size * 2][torch.eye(batch_size * 2, dtype=torch.bool, device=device)] = -1e9

        labels = torch.cat(
            [torch.arange(batch_size, batch_size * 2, device=device), torch.arange(batch_size, device=device)], dim=0
        )

        loss = self.ce(sim_mat, labels)

        return loss


@LOSSES.register_class
class TiCoLoss(Module):
    def __init__(self, beta: float = 0.9, rho: float = 20.0):
        super().__init__()

        self.beta = beta
        self.rho = rho

    def forward(self, prev_cov_matrix, emb_1, emb_2):
        z_1 = torch.nn.functional.normalize(emb_1, dim=-1)
        z_2 = torch.nn.functional.normalize(emb_2, dim=-1)
        B = torch.mm(z_1.T, z_1) / z_1.shape[0]
        prev_cov_matrix = self.beta * prev_cov_matrix + (1 - self.beta) * B
        loss = -(z_1 * z_2).sum(dim=1).mean() + self.rho * (torch.mm(z_1, prev_cov_matrix) * z_1).sum(dim=1).mean()
        return loss, prev_cov_matrix
