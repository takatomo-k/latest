import torch

class BaseModel(torch.nn.Module) :
    def __init__(self, *args, **kwargs) :
        super().__init__()
        pass

    @property
    def config(self) :
        return {'class':str(self.__class__)}
    pass
