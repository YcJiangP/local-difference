"""Block-normalized paired logistic heads, solved in the training row space."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


class BlockHeadPath:
    def transform(self, blocks):
        return np.concatenate([self.scalers[k].transform(np.asarray(blocks[k], float))/np.sqrt(self.dimensions[k])
                               for k in self.names], axis=1)

    @staticmethod
    def objective(v, a, b, y, weight, c):
        w, intercept = v[:-1], v[-1]
        z0, z1 = a@w+intercept, b@w+intercept
        n = len(y)
        r0, r1 = weight*(expit(z0)-y), weight*(expit(z1)-y)
        difference = a-b
        dz = difference@w
        loss = .5*np.mean(weight*(np.logaddexp(0,z0)-y*z0+np.logaddexp(0,z1)-y*z1))
        loss += .5*np.mean(weight*dz*dz)+.5/(c*n)*np.dot(w,w)
        grad = .5*(a.T@r0+b.T@r1)/n+difference.T@(weight*dz)/n+w/(c*n)
        return float(loss), np.r_[grad, .5*(r0.mean()+r1.mean())]

    def fit(self, original, augmented, y, cs):
        self.names = list(original)
        assert self.names == list(augmented)
        y = np.asarray(y, float)
        assert set(y)=={0.,1.}
        self.scalers = {k:StandardScaler().fit(np.asarray(original[k],float)) for k in self.names}
        self.dimensions = {k:original[k].shape[1] for k in self.names}
        x0, x1 = self.transform(original), self.transform(augmented)
        assert x0.shape == x1.shape and len(x0)==len(y) and np.isfinite(x0).all() and np.isfinite(x1).all()
        xx = np.concatenate([x0,x1])
        # The minimizer lies in the row span: loss and consistency depend on Xw,
        # and a component orthogonal to that span only increases the L2 penalty.
        gram = xx@xx.T
        eig, vectors = np.linalg.eigh(gram)
        cutoff = max(float(eig.max())*1e-12, 1e-14)
        keep = eig > cutoff
        roots, vectors = np.sqrt(eig[keep]), vectors[:,keep]
        z = vectors*roots
        n = len(y)
        counts = np.bincount(y.astype(int))
        weight = n/(2*counts[y.astype(int)])
        self.heads = {}
        initial = np.zeros(z.shape[1]+1)
        for c in sorted(cs):
            opt = minimize(lambda v:self.objective(v,z[:n],z[n:],y,weight,c),initial,
                           jac=True,method='L-BFGS-B',
                           options=dict(maxiter=1500,ftol=1e-13,gtol=1e-8,maxls=40))
            if not opt.success:
                raise ValueError('Paired logistic solve failed: '+str(opt.message))
            coef = xx.T@(vectors@(opt.x[:-1]/roots))
            raw = np.r_[coef,opt.x[-1]]
            loss, gradient = self.objective(raw,x0,x1,y,weight,c)
            assert abs(loss-opt.fun)<1e-7
            assert np.max(np.abs(xx@coef-z@opt.x[:-1]))<2e-6
            self.heads[float(c)] = dict(coef=coef,intercept=float(opt.x[-1]),
                                      diagnostics=dict(C=float(c),success=True,iterations=int(opt.nit),
                                          loss=float(loss),primal_gradient_max=float(np.abs(gradient).max()),
                                          rank=int(keep.sum()),eigen_cutoff=cutoff,training_n=n))
            initial = opt.x.copy()
        return self

    def logits(self, blocks, c):
        h = self.heads[float(c)]
        return self.transform(blocks)@h['coef']+h['intercept']
