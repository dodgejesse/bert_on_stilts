
from torch.utils.data import Sampler



class OrderedSampler(Sampler):
    def __init__(self, data_source, order_iterator):
        self.data_source = data_source
        self.order_iterator = order_iterator
        
    def __iter__(self):
        return self.order_iterator

    def __len__(self):
        return len(self.data_source)
