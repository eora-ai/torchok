import torch.nn as nn

from src.models.layers import ConvBnAct
from .utils import FeatureHooks


class BackboneBase(nn.Module):

    def create_hooks(self):
        self.stage_names = [i['module'] for i in self.feature_info]
        self.encoder_channels = [i['num_chs'] for i in self.feature_info]
        hooks = [dict(module=name, type='forward') for name in self.stage_names]
        self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def create_neck(self, set_neck):
        self.set_neck = set_neck
        if set_neck:
            # Neck (Self-Distillation)
            modules = []
            for in_c, out_c in zip(self.encoder_channels[:-1], self.encoder_channels[1:]):
                modules.append(ConvBnAct(in_c, out_c, kernel_size=3, stride=2))
            self.neck = nn.ModuleList(modules)
        else:
            self.neck = nn.Identity()

    def forward_features(self, x):
        raise NotImplementedError

    def forward_neck(self, x):
        if self.set_neck:
            for i, module in enumerate(self.neck):
                if x.size(1) == self.encoder_channels[i]:
                    x = module(x)
        return x

    def forward(self, x):
        # Return features for classification.
        y = self.forward_features(x)
        y = self.forward_neck(y)
        return y

    def forward_backbone_features(self, x):
        # Return intermediate features (for down-stream tasks).
        last_features = self.forward_features(x)
        backbone_features = self.feature_hooks.get_output(x.device)
        backbone_features = list(backbone_features.values())
        backbone_features = [x] + backbone_features
        return last_features, backbone_features

    def forward_stage_features(self, x):
        # Return intermediate features (for self-distillation).
        x = self.forward_features(x)
        return list(self.feature_hooks.get_output(x.device).values())

    def init_weights(self):
        # #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
