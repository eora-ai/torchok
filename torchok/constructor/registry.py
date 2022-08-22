import fnmatch
import sys
from collections import defaultdict
from typing import Callable
from typing import List, Union

from timm.models.registry import _natural_key


class Registry:
    """Registry of pipeline's components: models, datasets, metrics, etc.

    The registry is meant to be used as a decorator for any classes or function,
    so that they can be accessed by class name for instantiating. It also contains mapping from object name to model
    where this object stored and mapping from module to set of objects stored in module.
    Example:
        COMPONENTS = Registry('components')
        @COMPONENTS.register_class
        class SomeComponent:
            ...
    """

    def __init__(self, name: str):
        """
        Args:
            name: Component name.
        """
        self.name = name

        self.entrypoints = {}  # mapping of class/function names to entrypoint fns
        self.module_to_objects = defaultdict(set)  # dict of sets to check membership of class/function in module
        self.object_to_module = {}  # mapping of class/function names to its module name

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self.name}, items={list(self.entrypoints)})'
        return format_str

    def __contains__(self, item):
        return item in self.entrypoints

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key: str):
        """Search class type by class name.

        Args:
            key: Component class name.

        Returns:
            Found class type.

        Raises:
            KeyError: If key not in dictionary.
        """
        if key not in self.entrypoints:
            raise KeyError(f'{key} is not in the {self.name} registry')

        result = self.entrypoints[key]

        return result

    def register_class(self, fn: Callable):
        """Register a new entrypoint.

        Args:
            fn: function to be registered.

        Raises:
            TypeError: If fn is not callable.
            KeyError: If class_type is already registered.

        Returns:
            Input callable object.
        """
        if not callable(fn):
            raise TypeError(f'{fn} must be callable')
        class_name = fn.__name__
        if class_name in self.entrypoints:
            raise KeyError(f'{class_name} is already registered in {self.name}')

        mod = sys.modules[fn.__module__]
        module_name_split = fn.__module__.split('.')
        module_name = module_name_split[-1] if len(module_name_split) else ''

        # add model to __all__ in module
        model_name = fn.__name__
        if hasattr(mod, '__all__'):
            mod.__all__.append(model_name)
        else:
            mod.__all__ = [model_name]

        # add entries to registry dict/sets
        self.entrypoints[model_name] = fn
        self.object_to_module[model_name] = module_name
        self.module_to_objects[module_name].add(model_name)

        return fn

    def list_models(self, filter: str = '', module: str = '',
                    exclude_filters: Union[str, List[str]] = '') -> List[str]:
        """Filter stored objects by given criteria

        Args:
            filter: Wildcard filter string that works with fnmatch
            module: Limit model selection to a specific sub-module (ie 'gen_efficientnet')
            exclude_filters: Wildcard filters to exclude models after including them with filter

        Returns:
            Return list of filtered object/classes names, sorted alphabetically.

        Example:
            model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
            model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
        """
        if module:
            all_models = list(self.module_to_objects[module])
        else:
            all_models = self.entrypoints.keys()
        if filter:
            models = []
            include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
            for f in include_filters:
                include_models = fnmatch.filter(all_models, f)  # include these models
                if len(include_models):
                    models = set(models).union(include_models)
        else:
            models = all_models
        if exclude_filters:
            if not isinstance(exclude_filters, (tuple, list)):
                exclude_filters = [exclude_filters]
            for xf in exclude_filters:
                exclude_models = fnmatch.filter(models, xf)  # exclude these models
                if len(exclude_models):
                    models = set(models).difference(exclude_models)

        return list(sorted(models, key=_natural_key))
