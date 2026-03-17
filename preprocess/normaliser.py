
class BaseNormaliser:
    def fit(self, x): raise NotImplementedError
    def normalise(self, x): raise NotImplementedError
    def unnormalise(self, x): raise NotImplementedError
    def to_device(self, device): raise NotImplementedError

  
class ZScoreNormaliser(BaseNormaliser):
    def fit(self, x):
        self.mean = x.mean()
        self.std = x.std()
    def normalise(self, x): return (x - self.mean) / (self.std + 1e-8)
    def unnormalise(self, x): return x * self.std + self.mean
    def to_device(self, device): self.mean, self.std = self.mean.to(device), self.std.to(device)


class minmaxNormaliser(BaseNormaliser):
    def fit(self, x):
        self.max = x.max()
        self.min = x.min()
    def normalise(self, x): return (x-self.min) / (self.max - self.min)
    def unnormalise(self, x): return x * (self.max - self.min) + self.min
    def to_device(self, device): self.max, self.min = self.max.to(device), self.min.to(device)
    