## Support code for Bayesian MCMC
##
##
import numpy as np
from scipy import stats, spatial

class DataGenerator:

    """Generate data for simple prediction modelling"""

    def __init__(self, M=3, r=1, noise=0.2, randOffset=3):
        rs = randOffset
        self._States = {'TRAIN':0+rs, 'VALIDATION':1+rs, 'TEST':2+rs,}
        
        self._xmin = 0
        self._xmax = 10
        self._noiseStd = noise
        self._M = M
        self._r = r
        
        state = rs+1000 # Different state for generator
        wstd = 1
        self._Centres = np.linspace(0,1,self._M)*self._xmax
        self._RBF = RBFGenerator(self._Centres, width = self._r)
        self._W = stats.norm.rvs(size=(self._M,1), scale=wstd,
                                random_state=state)
        

    def _make_data(self, name, N, noiseStd=0):
        state = self._States[name]
        x = np.sort(stats.uniform.rvs(size=(N,1), random_state=state)
                    *self._xmax, axis=0)
        PHI = self._RBF.evaluate(x)
        y = np.dot(PHI,self._W)
        if noiseStd:
            t = y + stats.norm.rvs(size=(N,1), scale=noiseStd,
                                random_state=state)
        else:
            t = y
        return (x,t)
        

    def get_data(self, name, N):
        name = name.upper()
        if name=='TRAIN':
            return self._make_data(name, N, self._noiseStd)
        elif name=='VALIDATION':
            return self._make_data(name, N, self._noiseStd)
        elif name=='TEST':
            return self._make_data(name, N, 0)
        else:
            raise ValueError('Invalid data set name')
        

class RBFGenerator:

    """Generate Gausian RBF basis matrices"""

    def __init__(self, Centres, width=1):
        self._r = width
        self._M = len(Centres)
        self._Cent = Centres.reshape((self._M,1))

    def evaluate(self, X):
        N = len(X)
        PHI = np.empty((N, self._M))
        PHI = np.exp(-spatial.distance.cdist(X,self._Cent,
                                             metric="sqeuclidean")
            / (self._r**2))

        return PHI


## POSTERIOR
##
def compute_posterior(PHI,t,alph,s2):
    M       = PHI.shape[1]
    bet     = 1/s2
    SIGMA   = np.linalg.inv(bet*np.dot(PHI.T,PHI) + alph*np.eye(M))
    Mu      = bet*np.dot(SIGMA,np.dot(PHI.T,t))
    #
    return (Mu, SIGMA)


## LOG MARGINAL LIKELIHOOD
##
def compute_log_marginal(PHI,t,alph,s2):
    #
    (N, M)  = PHI.shape
    #
    # Compute using scipy.stats (not necessarily the best way)
    #
    C = s2*np.eye(N) + np.dot(PHI,PHI.T)/alph
    #
    lgp = stats.multivariate_normal.logpdf(t.T, mean=None, cov=C,
                                           allow_singular=True)
                                           #
    return lgp
