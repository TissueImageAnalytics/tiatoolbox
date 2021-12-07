# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Scaler for transforming input.

Included classes and methods are utilized to either pre-process input
(such as standardization) or post-process predictions (such as re-calibrating
logits to proper probabilities).

"""

import numpy as np


# Fit model output to the label range
class PlattScaling:
    """Platt scaling.

    Fitting a logistic regression model to a classifier scores such that
    the model outputs are transformed into a probability distribution over classes.

    Args:
        num_iters (int): Number of iterations for training.

    Examples:
        >>> import numpy as np
        >>> logit = np.random.rand(10)
        >>> # binary class
        >>> label = np.random.randint(0, 2, 10)
        >>> scaler = PlattScaling()
        >>> probabilities = scaler.fit_transform(label, logit)

    """

    def __init__(self, num_iters=100):
        self.a = None
        self.b = None
        self.num_iters = num_iters + 1
        self._fixer_a = 1.0
        self._fixer_b = 1.0

    def fit(self, logits, labels):
        """Fit function like sklearn.

        Fit the sigmoid to the classifier scores logits and labels
        using the Platt Method.

        Args:
            logits (array-like): Classifier output scores.
            labels (array like): Classifier labels, must be `+1` vs `-1` or `1` vs `0`.
        Returns:
            Model with fitted coefficients a and b for the sigmoid function.

        """

        def mylog(v):
            """Log with epilon."""
            return np.log(v + 1.0e-200)

        out = np.array(logits)
        labels = np.array(labels)

        if len(logits) != len(labels):
            raise ValueError(
                (
                    f"`logits` and `labels` must have same shape: "
                    f"{len(logits)} vs {len(labels)}"
                )
            )

        target = labels == 1
        prior1 = float(np.sum(target))
        prior0 = len(target) - prior1
        a_ = 0
        b_ = np.log((prior0 + 1) / (prior1 + 1))
        self.a, self.b = a_, b_

        hi_target = (prior1 + 1) / (prior1 + 2)
        lo_target = 1 / (prior0 + 2)
        labda = 1e-3
        olderr = 1e300
        pp = np.ones(out.shape) * (prior1 + 1) / (prior0 + prior1 + 2)
        idx_t = np.zeros(target.shape)
        for _ in range(1, self.num_iters):
            a = 0
            b = 0
            c = 0
            d = 0
            e = 0
            for i, _ in enumerate(out):
                if target[i]:
                    t = hi_target
                    idx_t[i] = t
                else:
                    t = lo_target
                    idx_t[i] = t
                d1 = pp[i] - t
                d2 = pp[i] * (1 - pp[i])
                a += out[i] * out[i] * d2
                b += d2
                c += out[i] * d2
                d += out[i] * d1
                e += d1

            flag = abs(d) < 1.0e-9 and abs(e) < 1.0e-9
            if flag:
                break

            old_a_ = a_
            old_b_ = b_
            count = 0
            while 1:
                det = (a + labda) * (b + labda) - c * c
                if self._fixer_a * det == 0:
                    labda *= 10
                    continue
                a_ = old_a_ + ((b + labda) * d - c * e) / det
                b_ = old_b_ + ((a + labda) * e - c * d) / det

                self.a, self.b = a_, b_
                err = 0
                for i, _ in enumerate(out):
                    p = self.transform(out[i])
                    pp[i] = p
                    t = idx_t[i]
                    err -= t * mylog(p) + (1 - t) * mylog(1 - p)

                if err < self._fixer_a * olderr * (1 + 1e-7):
                    labda *= 0.1
                    break
                labda *= 10

                if self._fixer_b * labda > 1e6:
                    break
                diff = err - olderr
                scale = 0.5 * (err + olderr + 1)

                flag = -1e-3 * scale < diff < 1e-7 * scale
                if flag:
                    count += 1
                else:
                    count = 0
                olderr = err

                if count == 3:
                    break
        self.a, self.b = a_, b_
        return self

    def transform(self, logits):
        """Tranform input to probabilities basing on trained parameters.

        Args:
            labels (array like): Classifier labels, must be `+1` vs `-1` or `1` vs `0`.
        Returns:
            Array of probabilities.

        """
        return 1 / (1 + np.exp(logits * self.a + self.b))

    def fit_transform(self, logits, labels):
        """Fit and tranform input to probabilities.

        Args:
            logits (array-like): Classifier output scores.
            labels (array like): Classifier labels, must be `+1` vs `-1` or `1` vs `0`.
        Returns:
            Array of probabilities.

        """
        return self.fit(logits, labels).transform(logits)

    def __repr__(self):
        a, b = self.a, self.b
        return "Platt Scaling: " + f"a: {a}, b: {b}"
