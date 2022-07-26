import fnmatch
import re
from typing import Callable
from collections import defaultdict
import sys

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


class Registry:
    """Registry of pipeline's components: models, datasets, metrics, etc.
    
    The registry is meant to be used as a decorator for any classes or function,
    so that they can be accessed by class name for instantiating. 
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
        self.__name = name

        self.__entrypoints = {}  # mapping of class/function names to entrypoint fns
        self.__module_to_objects = defaultdict(set)  # dict of sets to check membership of class/function in module
        self.__object_to_module = {}  # mapping of class/function names to module names

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self.__name}, items={list(self.__entrypoints)})'
        return format_str

    def __contains__(self, item):
        return item in self.__entrypoints

    def __getitem__(self, key):
        return self.get(key)

    @property
    def name(self):
        return self.__name

    @property
    def entrypoints(self):
        return self.__entrypoints

    @property
    def module_to_objects(self):
        return self.__module_to_objects

    @property
    def object_to_module(self):
        return self.__object_to_module

    def get(self, key: str):
        """Search class type by class name.

        Args:
            key: Component class name.

        Returns:
            Found class type.

        Raises:
            KeyError: If key not in dictionary.
        """
        if key not in self.__entrypoints:
            raise KeyError(f'{key} is not in the {self.__name} registry')

        result = self.__entrypoints[key]

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
        if class_name in self.__entrypoints:
            raise KeyError(f'{class_name} is already registered in {self.__name}')

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
        self.__entrypoints[model_name] = fn
        self.__object_to_module[model_name] = module_name
        self.__module_to_objects[module_name].add(model_name)

        return fn

    def list_models(self, filter='', module='', exclude_filters=''):
        """ Return list of available object/classes names, sorted alphabetically

        Args:
            filter (str) - Wildcard filter string that works with fnmatch
            module (str) - Limit model selection to a specific sub-module (ie 'gen_efficientnet')
            exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter

        Example:
            model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
            model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
        """
        if module:
            all_models = list(self.__module_to_objects[module])
        else:
            all_models = self.__entrypoints.keys()
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
