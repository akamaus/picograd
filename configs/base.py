class BaseConfig:
    """ Nothing, just knows how to assign keyword parameter values to attributes """
    def override_attrs(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                obj_v = getattr(self, k)
                if isinstance(obj_v, bool):
                    if v in "1 T t True true".split():
                        new_v = True
                    elif v in "0 F f False false".split():
                        new_v = False
                    else:
                        raise ValueError(f"Can't parse {k} = v")
                elif isinstance(v, str) and obj_v is not None:
                    new_v = getattr(self, k).__class__(v)
                else:
                    new_v = v
                setattr(self, k, new_v)
            else:
                raise ValueError(f'Unknown parameter "{k}"={v}')

        for k, _ in self.iter_attrs():
            if getattr(self, k) is None:
                raise ValueError(f'Unspecified parameter', k)

    def __repr__(self):
        args = ', '.join(map(lambda p: f'{p[0]}={p[1]}', self.iter_attrs()))
        return f'{self.__class__}({args})'

    def iter_attrs(self):
        for k in dir(self):
            v = getattr(self, k)
            if callable(v):
                continue

            if k[0] != '_' and k.islower():
                yield k, v
