class Registry:
    # !TODO may be we need to change the name of class to TorchokModule
    """ Class for generate Deep Learning modules.
    
    Each module can write to the register memory (dict) classes types that need to be remembered.
    
    Attributes:
        __name: Module name.
        __class_dict: Dictionary[str, type] - registr memory, where is the key - string of class name and 
            the value is class type. 
    """

    def __init__(self, name: str):
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
            key: Module class name.

        Returns:
            Found class type.
        """
        result = self.__class_dict.get(key, None)
        if result is None:
            raise KeyError(f'{key} is not in the {self.__name} registry')
        return result

    def __register_class(self, class_class: type):
        """Registrate a new class.

        Args:
            class_class: Class to be registered.
        """
        if not (isinstance(class_class, type) or callable(class_class)):
            raise TypeError(f'class must be a class, but got {type(class_class)}')
        class_name = class_class.__name__
        if class_name in self.__class_dict:
            raise KeyError(f'{class_name} is already registered in {self.__name}')
        self.__class_dict[class_name] = class_class

    def register_class(self, cls: type):
        """Registrate a new class.

        Args:
            cls: Class to be registered.
        """
        self.__register_class(cls)
        # !TODO is we really need to return some cls? 
        return cls
