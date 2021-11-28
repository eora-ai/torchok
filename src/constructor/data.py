from src.data import datasets
from src.data import transforms as module_transforms
from src.registry import DATASETS
from .config_structure import DatasetParams


def _prepare_transforms_recursive(transforms):
    transforms_list = []

    for transform_info in transforms:
        if isinstance(transform_info, dict):
            transform_name = transform_info['name']
            transform_params = transform_info.get('params', {})
        else:
            transform_name = transform_info.name
            transform_params = transform_info.params

        if transform_name == 'Compose':
            transform = prepare_compose(**transform_params)
        elif transform_name == 'OneOf':
            transform = prepare_oneof(**transform_params)
        else:
            transform = module_transforms.__dict__[transform_name](**transform_params)

        transforms_list.append(transform)

    return transforms_list


def prepare_compose(transforms, p=1.0):
    transforms_list = _prepare_transforms_recursive(transforms)
    transform = module_transforms.Compose(transforms_list, p=p)
    return transform


def prepare_oneof(transforms, p=0.5):
    transforms_list = _prepare_transforms_recursive(transforms)
    transform = module_transforms.OneOf(transforms_list, p=p)
    return transform


def create_transforms(transforms_params):
    if transforms_params is None:
        return None
    return prepare_compose(transforms_params)


def prepare_dataset(dataset_name, common_dataset_params, dataset_params):
    if dataset_name not in datasets.__dict__:
        raise ValueError("No such dataset in the library: {}".format(dataset_name))

    transform = create_transforms(dataset_params.transform)
    augment = create_transforms(dataset_params.augment)

    other_params = dataset_params.kwargs.copy()
    if common_dataset_params:
        other_params.update(common_dataset_params)

    dataset = datasets.__dict__[dataset_name](
        labels=dataset_params.labels, transform=transform, augment=augment,
        test_mode=dataset_params.test_mode, **other_params)

    return dataset


def create_dataset(dataset_name: str, common_dataset_params: dict, params: DatasetParams):
    transform = create_transforms(params.transform)
    augment = create_transforms(params.augment)

    params.params.update(common_dataset_params)

    dataset_class = DATASETS.get(dataset_name)
    return dataset_class(transform=transform, augment=augment, **params.params)
