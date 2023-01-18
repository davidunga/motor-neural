from impl.hatsopoulos import HatsoData
import geometrik as gk
from motor import KinFnc, preprocess_traj, derivative, cart2polar, KinData
from typechecking import *
import numpy as np
import matplotlib.pyplot as plt


class GeomKinFnc(KinFnc):

    def __init__(self, smooth_sig: float = .1):
        super().__init__()
        self.smooth_sig = smooth_sig

    def __call__(self, X: NpNx2[float], t: NpVec[float]) -> KinData:
        X, ut = preprocess_traj(X, t, self.smooth_sig)
        s0, s1, s2, k0, k1, k2 = gk.measures.arclens_and_curvatures(X)

        vel, acc, jrk = derivative(X, ut.t, n=3)
        spd0, = derivative(s0, ut.t)
        spd1, = derivative(s1, ut.t)

        kin = {'X': X,
               'velx': vel[:, 0],
               'vely': vel[:, 1],
               'accx': acc[:, 0],
               'accy': acc[:, 1],
               'crv0': k0,
               'crv1': k1,
               'crv2': k2,
               'spd0': spd0,
               'spd1': spd1,
               'spd2': np.linalg.norm(vel, axis=1),
               'acc2': np.linalg.norm(acc, axis=1)}

        return KinData(ut.fs, ut.t0, kin)


data = HatsoData.make("~/data/hatsopoulos", "CO_01", lag=.1, bin_sz=.01, kin_fnc=GeomKinFnc())
print(".")