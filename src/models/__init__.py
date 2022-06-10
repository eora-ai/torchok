from . import backbones
from . import base_model
from . import identity_model
from . import heads
from . import necks
from . import poolings


from src.constructor import HEADS, POOLINGS


HEADS.register_class(identity_model.Identity)
POOLINGS.register_class(identity_model.Identity)
