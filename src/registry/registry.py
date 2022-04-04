class Registry:

    def __init__(self, name):
        self._name = name
        self._class_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self._name}, items={list(self._class_dict)})'
        return format_str

    def __contains__(self, item):
        return item in self._class_dict

    def __getitem__(self, key):
        return self.get(key)

    @property
    def name(self):
        return self._name

    @property
    def class_dict(self):
        return self._class_dict

    def get(self, key):
        result = self._class_dict.get(key, None)
        if result is None:
            raise KeyError(f'{key} is not in the {self._name} registry')
        return result

    def _register_class(self, class_class):
        """Register a class.
        :param class_class: Class to be registered.
        """
        if not (isinstance(class_class, type) or callable(class_class)):
            raise TypeError(f'class must be a class, but got {type(class_class)}')
        class_name = class_class.__name__
        if class_name in self._class_dict:
            raise KeyError(f'{class_name} is already registered in {self.name}')
        self._class_dict[class_name] = class_class

    def register_class(self, cls):
        self._register_class(cls)
        return cls
