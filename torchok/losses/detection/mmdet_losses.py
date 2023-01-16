import types

from mmdet.models.builder import LOSSES as MMLOSSES

from torchok.constructor import LOSSES

for class_name, class_type in MMLOSSES.module_dict.items():
    if not class_name.endswith('Loss'):
        continue

    mm_class = types.new_class(f'MM{class_name}', bases=(class_type,))
    LOSSES.register_class(mm_class)
