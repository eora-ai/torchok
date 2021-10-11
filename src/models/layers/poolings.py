import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# from https://amaarora.github.io/2020/08/30/gempool.html
class GeM(nn.Module):
    def __init__(self, p=3):
        super(GeM, self).__init__()
        self.p = p
        self.eps = 1e-6

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return f'({self.__class__.__name__} p={self.p:.4f}, eps={self.eps})'


class RMAC(nn.Module):
    def __init__(self, L=3):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = 1e-6

    def forward(self, x):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + self.eps).expand_as(v)

        for l in range(1, self.L + 1):
            wl = math.floor(2 * w // (l + 1))
            wl2 = math.floor(wl // 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) // (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) // (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + self.eps).expand_as(vt)
                    v += vt

        return v


class CAC(nn.Module):
    def __init__(self):
        super(CAC, self).__init__()
        self.eps = 1e-6

    def forward(self, x):

        masks = self._create_circular_masks(x)

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + self.eps).expand_as(v)

        for mask in masks:
            C = torch.mul(x[:, :], mask)
            vt = F.max_pool2d(C, (C.size(-2), C.size(-1)))
            vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + self.eps).expand_as(vt)

            v += vt
        return v

    def _create_circular_masks(self, input):
        w = input.size(3)
        h = input.size(2)
        masks = []

        cen_w = w // 2
        cen_h = h // 2

        y = torch.arange(-cen_w, w - cen_w).view(-1, 1)
        x = torch.arange(-cen_h, h - cen_h).view(1, -1)

        y2 = y * y
        x2 = x * x

        start_mask = x2 + y2 <= 0

        masks.append(start_mask.type_as(input))

        max_r = torch.sqrt(torch.max(x2 + y2).double()).int() + 1

        for radius in range(1, int(max_r)):
            ring_masks = [
                (x2 + y * y > (radius - 1) ** 2) &
                (x2 + y2 <= radius ** 2) &
                (x >= 0) & (y >= 0),
                (x2 + y2 > (radius - 1) ** 2) &
                (x2 + y2 <= radius ** 2) &
                (x <= 0) & (y >= 0),
                (x2 + y2 > (radius - 1) ** 2) &
                (x2 + y2 <= radius ** 2) &
                (x >= 0) & (y <= 0),
                (x2 + y2 > (radius - 1) ** 2) &
                (x2 + y2 <= radius ** 2) &
                (x <= 0) & (y <= 0)
            ]
            circle_masks = [
                (x2 + y2 <= radius ** 2) & (x >= 0) & (y >= 0),
                (x2 + y2 <= radius ** 2) & (x <= 0) & (y >= 0),
                (x2 + y2 <= radius ** 2) & (x >= 0) & (y <= 0),
                (x2 + y2 <= radius ** 2) & (x <= 0) & (y <= 0)
            ]

            for ring_mask, circle_mask in zip(ring_masks, circle_masks):
                masks.append(ring_mask.type_as(input))
                masks.append(circle_mask.type_as(input))

        return masks


# https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Babenko_Aggregating_Local_Deep_ICCV_2015_paper.pdf
class SPoC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[-2:]
        sigma = min(h, w) / 3.

        xs = torch.arange(w, device=x.device, dtype=x.dtype)
        ys = torch.arange(h, device=x.device, dtype=x.dtype)[:, None]

        xs = (xs - (w - 1) / 2) ** 2
        ys = (ys - (h - 1) / 2) ** 2
        gauss_weight = torch.exp(-(xs + ys) / (2 * sigma ** 2))

        x = (x * gauss_weight).sum(dim=(2, 3))

        return x
