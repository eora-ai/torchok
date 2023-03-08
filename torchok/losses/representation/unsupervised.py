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
class TiCoLoss(torch.nn.Module):
    """Implementation of the Tico Loss from Tico[0] paper.
    This implementation takes inspiration from the code published
    by sayannag using Lightly. [1]
    [0] Jiachen Zhu et. al, 2022, Tico... https://arxiv.org/abs/2206.10698
    [1] https://github.com/sayannag/TiCo-pytorch
    Attributes:
        Args:
            beta:
                Coefficient for the EMA update of the covariance
                Defaults to 0.9 [0].
            rho:
                Weight for the covariance term of the loss
                Defaults to 20.0 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
    Examples:
        >>> # initialize loss function
        >>> loss_fn = TiCoLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self,
        beta: float = 0.9,
        rho: float = 20.0,
        gather_distributed: bool = False,
        update_covariance_matrix: bool = True,
    ):
        super(TiCoLoss, self).__init__()
        self.beta = beta
        self.rho = rho
        self.C = None
        self.gather_distributed = gather_distributed
        self.update_covariance_matrix = update_covariance_matrix

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        """Tico Loss computation. It maximize the agreement among embeddings of different distorted versions of the same image
        while avoiding collapse using Covariance matrix.
        Args:
            z_a:
                Tensor of shape [batch_size, num_features=256]. Output of the learned backbone.
            z_b:
                Tensor of shape [batch_size, num_features=256]. Output of the momentum updated backbone.
            update_covariance_matrix:
                Parameter to update the covariance matrix at each iteration.
        Returns:
            The loss.
        """

        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert z_a.shape == z_b.shape, f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # normalize image
        z_a = torch.nn.functional.normalize(z_a, dim=1)
        z_b = torch.nn.functional.normalize(z_b, dim=1)

        # compute auxiliary matrix B
        B = torch.mm(z_a.T, z_a) / z_a.shape[0]

        # init covariance matrix
        if self.C is None:
            self.C = B.new_zeros(B.shape).detach()

        # compute loss
        C = self.beta * self.C + (1 - self.beta) * B
        loss = 1 - (z_a * z_b).sum(dim=1).mean() + self.rho * (torch.mm(z_a, C) * z_a).sum(dim=1).mean()

        # update covariance matrix
        if self.update_covariance_matrix:
            self.C = C.detach()

        return loss
