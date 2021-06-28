import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def check_matrix(Yt, Yc, S12, S23, T12, T13, T23, S2, S3):
    MFC = 1 / Yc * ((Yc / 2 / S23)**2 - 1) * (S2 + S3) + 1 / 4 / S23**2 * (S2 + S3)**2 + 1 / S23**2 * (T23**2 - S2 * S3) + 1 / S12**2 * (T12**2 + T13**2)
    MFT = (S2 + S3)**2 / Yt**2 + (T12**2 + T13**2) / S12**2 + (T23**2 - S2 * S3) / S23**2
    check = True
    if MFC >= 1 or MFT >= 1:
        check = False
    return check
def check_fibers(Xt, Xc, S12, S1, T12, T13):
    check = True
    if S1 >= 0:
        FF = (S1 / Xt)**2 + 1 / S12**2 * (T12**2 + T13**3)

    elif S1 < 0:
        FF = - S1 / Xc

    if FF >= 1:
        check = False
    return check


