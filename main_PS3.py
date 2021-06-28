import laminate_class
from numpy import  sin, cos, pi
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

stack = np.array([45, 90, -45, 0])
stack = np.hstack([stack, stack, stack])
stack = np.hstack([stack, np.flip(stack)])

specimen = laminate_class.laminate(stack_init=stack)

# determine laminate strength
running = True
Nx = -1
while running:
    F = np.array([Nx, 0, 0, 0, 0, 0])
    matrix, fibre = specimen.hashin_fail(F=F)
    # print(Nx)

    if matrix.any() == False or fibre.all() == False:
        running=False
    else:
        Nx -= 1

sigma_f = - Nx / specimen.h
print('Sigma_f = %1.3f MPa' % sigma_f)

# now onto impact properties
a = 150 #mm
b = 150 #mm

## first, experimentally determine k_ind for the given composite plate

E_imp_exp = 25e3 #mJ
D_ind = 25.4 #mm
d_w_exp = 1 #mm



#### laminate D matrix entries
D11 = specimen.D[0,0]
D12 = specimen.D[0,1]
D66 = specimen.D[2,2]
D22 = specimen.D[1,1]


running = True
F = 10 #N, initial value; final value found by iterative solver
E_tot = 0
while running:
    if round(E_tot * 100 / E_imp_exp,2) % 5 == 0:
        print('k_ind: %1.2f %% completed' % (E_tot * 100 / E_imp_exp))
    F_old = F
    w_max = 0
    for m in range(1,25):
        for n in range(1,25):
            num = (4 * F / a / b * sin(m * pi / 2)**2 * sin(n * pi / 2)**2)
            den = D11 * (m * pi / a)**4 + 2 * (D12 + 2 * D66) * (m**2 * n**2 * pi**4 / a**2 / b**2) + D22 * (n * pi / b)**4
            w_max += num / den
    U_p = 0.5 * F * w_max
    k_ind = F / d_w_exp**(3/2)
    E_ind = 2 / 5 * k_ind * d_w_exp**(5/2)

    E_tot = E_ind + U_p

    if E_tot <= E_imp_exp:
        F += 10
    else:
        running = False
        print('k_ind = %1.2f [N/mm^(3/2)]' % k_ind)
        print('F_peak = %1.2f N' % F)


k = k_ind


running = True
F = 10 #N, initial value; final value found by iterative solver
d_max = 0
while running:
    if round(d_max * 100 / (specimen.h / 2),1) % 5 == 0:
        print('h/2 dent: %1.2f %% completed' % (d_max * 100 / specimen.h / 2))
    F_old = F
    w_max = 0
    for m in range(1,25):
        for n in range(1,25):
            num = (4 * F / a / b * sin(m * pi / 2)**2 * sin(n * pi / 2)**2)
            den = D11 * (m * pi / a)**4 + 2 * (D12 + 2 * D66) * (m**2 * n**2 * pi**4 / a**2 / b**2) + D22 * (n * pi / b)**4
            w_max += num / den

    U_p = 0.5 * F * w_max
    d_max = (F / k) ** (2 / 3)
    E_ind = 2 / 5 * k * d_max**(5/2)

    E_tot = E_ind + U_p

    if d_max <= specimen.h/2:
        F += 100

    elif d_max > specimen.h/2:
        running = False
        print('E_tot = %1.2f J' % (E_tot / 1000))
        print('F_peak = %1.2f N' % F)


running = True
F = 10 #N
while running:
    d_max = (F / k) ** (2 / 3)
    R = D_ind / 2
    Rc = np.sqrt(R ** 2 - (R - d_max) ** 2)
    A = 2 * pi * Rc * specimen.h
    tau_avg = F / A
    if round(tau_avg / laminate_class.p.S * 100,1) % 5 == 0:
        print('Penetration: %1.2f %% completed' % (tau_avg / laminate_class.p.S * 100))

    if tau_avg < laminate_class.p.S:
        F += 1
    else:
        w_max = 0
        for m in range(1, 25):
            for n in range(1, 25):
                num = (4 * F / a / b * sin(m * pi / 2) ** 2 * sin(n * pi / 2) ** 2)
                den = D11 * (m * pi / a) ** 4 + 2 * (D12 + 2 * D66) * (
                            m ** 2 * n ** 2 * pi ** 4 / a ** 2 / b ** 2) + D22 * (n * pi / b) ** 4
                w_max += num / den
        U_p = 0.5 * F * w_max
        E_ind = 2 / 5 * k * d_max ** (5 / 2)
        E_tot = E_ind + U_p
        running = False
        print('F_pen = %1.2f N' % F)
        print('E = %1.2f J' % (E_tot / 1000))
        print('delta_max = %1.4f mm' % d_max)
        print('Rc = %1.4f mm' % Rc)

print('Units conversion')
print('E_tot = %1.2f ft-lb' % (0.737562149 * E_tot / 1000))
print('Normalised impact energy = %1.2f' % (0.737562149 * E_tot / 1000 / (specimen.h / 25.4)))

# R_arr = np.linspace(0.01, 20, 100)
# d0 = 3.8
# Rs = np.empty(0)
# FFS = np.empty(0)
# for R in R_arr:
#     running = True
#     sigma0 = sigma_f / 3 # starting condition, could be anything
#     while running:
#         y = R + d0
#         RHS = 1 + 0.5 * (R / y)**2 + 1.5 * (R / y)**4
#         sigmax = sigma0 * RHS
#
#         if sigmax >= sigma_f:
#             running = False
#             # print(sigmax, sigma_f, sigma0)
#             Rs = np.append(Rs, R)
#             FFS = np.append(FFS, sigma0)
#
#         else:
#             sigma0 += 0.1

# R = Rc
d0 = 3.8
# y = R + d0
# RHS = 1 + 0.5 * (R / y)**2 + 1.5 * (R / y)**4
den = d0 - (Rc**2 / 2) * (1 / (Rc + d0) - 1 / Rc) - (Rc**4 / 2) * (1 / (Rc + d0)**3 - 1 / Rc**3)
kt = 3 * (2 + (1 - (2 * Rc / b))**3) / (3 * (1 - 2*Rc/b))
CAI = 1 / den * d0 * 3 / kt
print('CAI inf = %1.4f' % (CAI * kt / 3))
print('CAI = %1.4f' %CAI)
print('sigma_o = %1.3f' % (CAI * sigma_f))


# img = plt.imread("CAI_Energy.png")
# fig1, ax1 = plt.subplots()
#
# ax1.imshow(img)
# ax1.axis('off')
# ax1.scatter(1937, 957, label = 'Current Work', color = 'orange', s=150)
# ax1.legend(fontsize=15, loc= (0.855, 0.485))

# sns.set()
# fig, ax = plt.subplots()
#
# ax.plot(Rs, FFS / sigma_f, label = 'OHC Strength - Whitney-Nuismer')
# ax.scatter(find_nearest(Rs, Rc), FFS[Rs == find_nearest(Rs, Rc)] / sigma_f, marker = '*', color= 'r', label = r'$R = R_c$')
# ax.set_xlabel(r'Hole radius $R$ [mm]')
# ax.set_ylabel(r'Normalised CAI strength $\frac{\sigma_0}{\sigma_f}$ [-]')
# ax.legend()
# ax.minorticks_on()
# ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.show()

print('Point on graph')
print('Normalised impact energy [ft-lb/in] = %1.2f' % (0.737562149 * E_tot / 1000 / (specimen.h / 25.4)))
print('CAI Normalised [-] = %1.2f' % CAI)