

import numpy as np


# Fit model output to the label range
class PlattScaling:
    def __init__(self):
        self.A = None
        self.B = None

    def fit(self, L, V):
        """Fit function like sklearn.

        Fit the sigmoid to the classifier scores V and labels L using the Platt Method.

        Args:  
            L (array like): Classifier labels (+1/-1 pr +1/0).
            V (array-like): Classifier output scores.
        Returns:
            Coefficients A and B for the sigmoid function

        """
        def mylog(v):
            if v == 0:
                return -200
            else:
                return np.log(v)

        out = np.array(V)
        L = np.array(L)
        assert len(V) == len(L)
        target = L == 1
        prior1 = float(np.sum(target))
        prior0 = len(target) - prior1
        A = 0
        B = np.log((prior0 + 1) / (prior1 + 1))
        self.A, self.B = A, B
        hiTarget = (prior1 + 1) / (prior1 + 2)
        loTarget = 1 / (prior0 + 2)
        labda = 1e-3
        olderr = 1e300
        pp = np.ones(out.shape) * (prior1 + 1) / (prior0 + prior1 + 2)
        T = np.zeros(target.shape)
        for it in range(1, 100):
            a = 0
            b = 0
            c = 0
            d = 0
            e = 0
            for i in range(len(out)):
                if target[i]:
                    t = hiTarget
                    T[i] = t
                else:
                    t = loTarget
                    T[i] = t
                d1 = pp[i] - t
                d2 = pp[i] * (1 - pp[i])
                a += out[i] * out[i] * d2
                b += d2
                c += out[i] * d2
                d += out[i] * d1
                e += d1
            if (abs(d) < 1e-9 and abs(e) < 1e-9):
                break
            oldA = A
            oldB = B
            count = 0
            while 1:
                det = (a + labda) * (b + labda) - c * c
                if det == 0:
                    labda *= 10
                    continue
                A = oldA + ((b + labda) * d - c * e) / det
                B = oldB + ((a + labda) * e - c * d) / det
                self.A, self.B = A, B
                err = 0
                for i in range(len(out)):
                    p = self.transform(out[i])
                    pp[i] = p
                    t = T[i]
                    err -= t * mylog(p) + (1 - t) * mylog(1 - p)
                if err < olderr * (1 + 1e-7):
                    labda *= 0.1
                    break
                labda *= 10
                if labda > 1e6:
                    break
                diff = err - olderr
                scale = 0.5 * (err + olderr + 1)
                if diff > -1e-3 * scale and diff < 1e-7 * scale:
                    count += 1
                else:
                    count = 0
                olderr = err
                if count == 3:
                    break
        self.A, self.B = A, B
        return self

    def transform(self, V):
        return 1 / (1 + np.exp(V * self.A + self.B))

    def fit_transform(self, L, V):
        return self.fit(L, V).transform(V)

    def __repr__(self):
        A, B = self.A, self.B
        return "Platt Scaling: " + f'A: {A}, B: {B}'
