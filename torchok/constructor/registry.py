class Registry:
    """Registry of pipeline's components: models, datasets, metrics, etc.
    
    The registry is meant to be used as a decorator for any classes, 
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
        self.__class_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self.__name}, items={list(self.__class_dict)})'
        return format_str

    def __contains__(self, item):
        return item in self.__class_dict

    def __getitem__(self, key):
        return self.get(key)

    @property
    def name(self):
        return self.__name

    @property
    def class_dict(self):
        return self.__class_dict

    def get(self, key: str):
        """Search class type by class name.

        Args:
            key: Component class name.

        Returns:
            Found class type.

        Raises:
            KeyError: If key not in dictionary.
        """
        if key not in self.__class_dict:
            raise KeyError(f'{key} is not in the {self.__name} registry')
        
        result = self.__class_dict[key]

        return result

    def register_class(self, class_type: type):
        """Register a new class.

        Args:
            class_type: Class to be registered.

        Raises:
            TypeError: If class_type not type.
            KeyError: If class_type already registered.

        Returns:
            Input class type.
        """
        if not (isinstance(class_type, type) or callable(class_type)):
            raise TypeError(f'class_type must be class type, but got {type(class_type)}')
        class_name = class_type.__name__
        if class_name in self.__class_dict:
            raise KeyError(f'{class_name} is already registered in {self.__name}')
        self.__class_dict[class_name] = class_type
        return class_type
