import numpy as np
### elastic
Ex = 143.4e3
Ey = 11.37e3
Gxy = 5.17e3
vxy = 0.32
vyx = vxy * Ey / Ex
vxz = vxy
vyz = 0.4

t_ply = 0.1524
rho = 1610
### Q matrix
Q = np.zeros((3, 3))
Qxx = Ex / (1 - vxy * vyx)
Qyy = Ey / (1 - vxy * vyx)
Qxy = vxy * Ey / (1 - vxy * vyx)

Q = np.array([[Qxx, Qxy, 0],
              [Qxy, Qyy, 0],
              [0, 0, Gxy]])
### Strength
Xt = 2510
Yt = 114
Xc = 1723
Yc = 302
S = 132

### PS3 for verification
# ### elastic
# Ex = 127.6e3
# Ey = 9.31e3
# Gxy = 5.38e3
# vxy = 0.28
# vyx = vxy * Ey / Ex
# t_ply = 0.3048
# rho = 1610
# ### Q matrix
# Q = np.zeros((3, 3))
# Qxx = Ex / (1 - vxy * vyx)
# Qyy = Ey / (1 - vxy * vyx)
# Qxy = vxy * Ey / (1 - vxy * vyx)
#
# Q = np.array([[Qxx, Qxy, 0],
#               [Qxy, Qyy, 0],
#               [0, 0, Gxy]])
# ### Strength
# Xt = 1784
# Yt = 35
# Xc = 1181.6
# Yc = 138.6
# S = 98.8
# # S_bear = 650

