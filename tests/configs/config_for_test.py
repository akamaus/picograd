from configs.base import BaseConfig


class Config(BaseConfig):
    def __init__(self, **kwargs):
        self.a = 2
        self.b = None
        super().__init__(**kwargs)
