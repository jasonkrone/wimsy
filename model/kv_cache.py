import torch


class KVCache(object):

    def __init__(self):
        self.k = None
        self.v = None

    def add(self, k, v):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)

    def seqlen(self):
        length = 0
        if self.k is not None:
            length = self.k.shape[2]
        return length

    def clear(self):
        self.k = None
        self.v = None


class MultiLayerCache(object):

    def __init__(self, num_layers):
        self.caches = [KVCache() for _ in range(num_layers)]

    def get_kv_cache_for_layer(self, i):
        return self.caches[i]

    def clear(self):
        for cache in self.caches:
            cache.clear()

