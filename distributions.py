from numpy import concatenate as cat
from numpy.random import multinomial, uniform, normal
from numpy import ravel

class Distribution(object):
    def __init__(self):
        pass
    
    def sample(self, N):
        pass
    
    def likelihood(self, p):
        pass

class Factor(Distribution):
    def __init__(self, *dists):
        for dist in ravel(dists):
            assert isinstance(dist, Distribution)
        self.dists = ravel(dists)
        self.len = sum([dist.len for dist in dists])
        self.num_params = sum([dist.num_params for dist in dists])
        
    def sample(self, N):
        return cat([dist.sample(N) for dist in self.dists], axis=1)
        

class Categorical(Distribution):
    def __init__(self, K=2):
        self.K = K
        self.len = K
        self.num_params = K
     
    def sample(self, N):
        return multinomial(1, [1/self.K]*self.K, size = N);
    

class Normal(Distribution):
    def __init__(self, mu=0, std=1):
        self.mu = mu
        self.std = std
        self.len = 1
        self.num_params = 2
     
    def sample(self, N):
        return normal(self.mu, self.std, size = (N,1));
        
    
class Uniform(Distribution):
    def __init__(self, low=-1, high=1):
        self.low = low
        self.high = high
        self.len = 1
        self.num_params = 2
     
    def sample(self, N):
        return uniform(self.low, self.high, size = (N,1));