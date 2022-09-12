from mmdet.models.necks import (BFP, ChannelMapper, CTResNetNeck, DilatedEncoder, DyHead, FPG, FPN_CARAFE, HRFPN,
                                NASFCOS_FPN, NASFPN, PAFPN, RFP, SSDNeck, YOLOV3Neck, YOLOXPAFPN)

from torchok.constructor import DETECTION_NECKS

DETECTION_NECKS.register_class(BFP)
DETECTION_NECKS.register_class(ChannelMapper)
DETECTION_NECKS.register_class(HRFPN)
DETECTION_NECKS.register_class(NASFPN)
DETECTION_NECKS.register_class(FPN_CARAFE)
DETECTION_NECKS.register_class(PAFPN)
DETECTION_NECKS.register_class(NASFCOS_FPN)
DETECTION_NECKS.register_class(YOLOV3Neck)
DETECTION_NECKS.register_class(DilatedEncoder)
DETECTION_NECKS.register_class(CTResNetNeck)
DETECTION_NECKS.register_class(RFP)
DETECTION_NECKS.register_class(FPG)
DETECTION_NECKS.register_class(SSDNeck)
DETECTION_NECKS.register_class(YOLOXPAFPN)
DETECTION_NECKS.register_class(DyHead)
