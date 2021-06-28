import numpy as np
from numpy import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
import seaborn as sns
import laminate_class
import impactor as imp
from matplotlib import cm
sns.set()

delta_x = 0.0025 #mm

### problems (a), (b), (c) and (d)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

plot_shear = True
plot_delaminations = True
plot_interface_stresses = True
plot_truncated_stresses = True

## selected laminate: [45, 90, -45, 0]_3s - fifth one
stack = np.array([45, 90, -45, 0])
stack = np.hstack([stack, stack, stack])
stack = np.hstack([stack, np.flip(stack)])
print('n-plies = %1.0f' % (stack.shape))
laminate = laminate_class.laminate(stack_init=stack)

# determine laminate strength
running = True
Nx = -1
while running:
    F = np.array([Nx, 0, 0, 0, 0, 0])
    matrix, fibre = laminate.hashin_fail(F=F)
    # print(Nx)

    if matrix.any() == False or fibre.all() == False:
        running=False
    else:
        Nx -= 1

sigma_f = - Nx / laminate.h
print('Sigma_f = %1.3f MPa' % sigma_f)
#### laminate D matrix entries
D11 = laminate.D[0,0]
D12 = laminate.D[0,1]
D66 = laminate.D[2,2]
D22 = laminate.D[1,1]


delta_max = 2
k_ind = 76400 #N and mm
F = k_ind * delta_max ** (3 / 2)
Rc = sqrt(imp.R**2 - (imp.R - delta_max)**2)

r1 = np.linspace(0, Rc, 150)
r2 = np.linspace(Rc, 80, 150)
r = np.hstack([r1, r2])
z_h = np.linspace(0,1, 50)

t_avg_1 = F * r1 / (2 * pi * laminate.h * Rc**2)
t_avg_2 = F / (2 * pi * r2 * laminate.h)

tau_avg = np.hstack([t_avg_1, t_avg_2])

tau_zh = np.zeros((z_h.shape[0], tau_avg.shape[0]))
tau_avg_plane = np.copy(tau_zh)
for i in range(tau_avg.shape[0]):
    tau_zh[:, i] = tau_avg[i] * (-6 * z_h ** 2 + 6 * z_h)
    tau_avg_plane[:, i] = tau_avg[i]

r_mesh, z_h_mesh = np.meshgrid(r,z_h)

if plot_shear:
    fig, ax = plt.subplots()

    ax.plot(r, tau_avg, label = 'Avg criterion')
    ax.plot(r, np.amax(tau_zh, axis = 0), label = 'Quadratic max')

    ax.set_xlabel(r'Distance from hole centre, $r$, [mm]')
    ax.set_ylabel(r'Shear stress, $\tau$, [MPa]')
    ax.legend()
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.plot_surface(r_mesh, z_h_mesh, tau_avg_plane, linewidth=0, cmap=cm.RdBu_r)
    ax1.plot_surface(r_mesh, z_h_mesh, tau_zh, cmap=cm.RdBu_r, linewidth = 0)
    ax1.set_xlabel(r'$r$, [mm]')
    ax1.set_ylabel(r'$\frac{z}{h}$, [-]')
    ax1.set_zlabel(r'$\tau$, [MPa]')
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

###############################################################################################################

# now moving to part (c)
## E max is 58.57 J
delta_s_max = np.linspace(0.01, 0.4, 20)
a, b = (150, 150)
E_norms = {}
E_tots = {}
# ply_surf_locs = np.linspace(0, 1, stack.shape[0]+1)  ## normalised
z_h = np.linspace(0,1, stack.shape[0]+1)
delamins_dict = {}
stresses_dict = {}
for delta_max in delta_s_max:
    delaminations = np.zeros((z_h.shape[0], r.shape[0]))
    F = k_ind * delta_max ** (3 / 2)
    w_max = 0
    for m in range(1,25):
        for n in range(1,25):
            num = (4 * F / a / b * sin(m * pi / 2)**2 * sin(n * pi / 2)**2)
            den = D11 * (m * pi / a)**4 + 2 * (D12 + 2 * D66) * (m**2 * n**2 * pi**4 / a**2 / b**2) + D22 * (n * pi / b)**4
            w_max += num / den
    U_p = 0.5 * F * w_max
    E_ind = 2 / 5 * k_ind * delta_max**(5/2)
    E_tot = E_ind + U_p
    E_norm = 0.737562149 * E_tot / 1000 / (laminate.h / 25.4)
    print('================================')
    print('delta_max = %1.4f' %delta_max)
    print('E_tot = %1.3f' % (E_tot / 1000))
    print('E_norm = %1.3f' % E_norm)
    Rc = sqrt(imp.R ** 2 - (imp.R - delta_max)**2)
    r1 = np.linspace(0, Rc, int(Rc // delta_x))
    r2 = np.linspace(Rc, 100, int((100 - Rc) // delta_x))
    r = np.hstack([r1, r2])
    t_avg_1 = F * r1 / (2 * pi * laminate.h * Rc ** 2)
    t_avg_2 = F / (2 * pi * r2 * laminate.h)
    tau_avg = np.hstack([t_avg_1, t_avg_2])
    tau_zh = np.zeros((z_h.shape[0], tau_avg.shape[0]))
    # tau_avg_plane = np.copy(tau_zh)
    for i in range(tau_avg.shape[0]):
        tau_zh[:, i] = tau_avg[i] * (-6 * z_h ** 2 + 6 * z_h)
        for j in range(tau_zh.shape[0]):
            if tau_zh[j,i] >= laminate_class.p.S:
                delaminations[j,i] = 1
    delamins_dict[delta_max] = delaminations
    stresses_dict[delta_max] = tau_zh
    E_norms[delta_max] = E_norm
    E_tots[delta_max] = E_tot

if plot_delaminations:
    fig_del, ax_del = plt.subplots()

    delamination = delamins_dict[delta_max] ## using last value of delta, can be changed
    for i in range(delamination.shape[0]-1):
        x = delamination[i, np.nonzero(delamination[i, :])] * r[np.nonzero(delamination[i, :])]
        y = delamination[i, np.nonzero(delamination[i, :])] * z_h[i]
        ax_del.plot(x[0],y[0], color = 'b')
        ax_del.plot(-np.flip(x[0]), y[0], color='b')
    i += 1
    x = delamination[i, np.nonzero(delamination[i, :])] * r[np.nonzero(delamination[i, :])]
    y = delamination[i, np.nonzero(delamination[i, :])] * z_h[i]
    ax_del.plot(-np.flip(x[0]), y[0], color='b')
    ax_del.plot(x[0], y[0], color='b', label = r'$\tau \geq S$')
    ax_del.set_xlabel(r'Distance from hole centre, $r$, [mm]')
    ax_del.set_ylabel(r'Normalised ply interface location $\frac{z}{h}$ [-]')
    ax_del.legend()
    ax_del.minorticks_on()
    ax_del.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax_del.set_title(r'Ply interfaces where $\tau \geq S$, E = %1.2f J' %(E_tot / 1000))
    ax_del.set_ylim(0,1)

if plot_interface_stresses:
    fig_int, ax_int = plt.subplots()
    tau = stresses_dict[delta_max]
    for i in range(1,tau.shape[0]//2+1):
        ax_int.plot(r, tau[i,:], label = r'$\frac{z}{h}$ = %1.3f' %z_h[i])
    ax_int.plot(r, np.ones_like(r) * laminate_class.p.S, label = 'S')
    ax_int.set_xlabel(r'$r$, [mm]')
    ax_int.set_ylabel(r'$\tau$ [MPa]')
    ax_int.legend(ncol = 4)
    ax_int.minorticks_on()
    ax_int.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # ax_int.set_title(r'Ply interfaces where $\tau \geq S$, E = %1.2f J' %(E_tot / 1000))

### now compute delamination size

index = delta_max
taus = stresses_dict[index]
taus_trunc = np.copy(taus)
delamination_s = delamins_dict[index]
for i in range(1,delamination.shape[0]-1):
    tau = taus[i,:]
    ## truncate stresses
    taus_trunc[i, :][taus_trunc[i, :] > laminate_class.p.S] = laminate_class.p.S
    delamination = delamination_s[i,:]
    A = np.trapz(tau[tau >= laminate_class.p.S] - laminate_class.p.S, r[tau >= laminate_class.p.S])
    # print(A)
    ## now, redistribute area and increase stresses
    if A > 0:
        ## calculate max area possible to the left:
        S_temp = np.ones_like(tau[:np.nonzero(delamination)[0][0]]) * laminate_class.p.S
        r_temp = r[:np.nonzero(delamination)[0][0]]
        max_left = np.trapz(S_temp, r_temp) - np.trapz(tau[:np.nonzero(delamination)[0][0]], r_temp)
        if A <= max_left:
            j = 1  ## initialise with first non-zero value of r
            running = True
            while running:
                r_temp = r[:j+1]
                r_temp_max = r_temp[-1]
                tau_temp = tau[:j+1]
                tau_temp_max = tau_temp[-1]
                if tau_temp_max >= laminate_class.p.S:
                    tau_temp_max = laminate_class.p.S
                Area_left_temp = tau_temp_max * r_temp_max
                Area_below = np.trapz(tau_temp, r_temp)
                difference = Area_left_temp - Area_below
                if difference >= A:
                    running = False
                    taus_trunc[i, :j] = tau_temp_max
                else:
                    j += 1
        elif A > max_left:
            start_delamination = np.nonzero(delamination)[0][0]
            end_delamination = np.nonzero(delamination)[0][-1]
            taus_trunc[i, :start_delamination] = laminate_class.p.S
            tau_temp = taus[i, :start_delamination]
            tau_temp_max = tau_temp[-1]
            r_temp = r[:start_delamination]
            r_temp_max = r_temp[-1]
            Area_left_temp = tau_temp_max * r_temp_max
            Area_below = np.trapz(tau_temp, r_temp)
            difference = Area_left_temp - Area_below
            A_differece = A - difference
            ## now determine d
            running = True
            j = end_delamination
            while running:
                r_temp = r[end_delamination - 1:j+1]
                tau_temp = taus[i, end_delamination - 1: j+1]
                Area_below = np.trapz(tau_temp, r_temp)
                Area_right_temp = laminate_class.p.S * (r_temp[-1] - r_temp[0])
                A_diff = Area_right_temp - Area_below

                if A_diff < A_differece:
                    j += 1
                else:
                    running = False
                    taus_trunc[i, end_delamination:j] = laminate_class.p.S
                if j > 250:
                    running = False
    delamination[taus_trunc[i, :] == laminate_class.p.S] = 1

if plot_truncated_stresses:
    fig_trunc, ax_trunc = plt.subplots()
    tau = taus_trunc
    for i in range(1,tau.shape[0]//2+1):
        ax_trunc.plot(r, tau[i,:], label = r'$\frac{z}{h}$ = %1.3f' %z_h[i])
    ax_trunc.plot(r, np.ones_like(r) * laminate_class.p.S, label = 'S')
    ax_trunc.set_xlabel(r'$r$, [mm]')
    ax_trunc.set_ylabel(r'$\tau$ [MPa]')
    ax_trunc.legend(ncol = 4)
    ax_trunc.minorticks_on()
    ax_trunc.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

if plot_delaminations:
    fig_del1, ax_del1 = plt.subplots()

    delamination = delamins_dict[delta_max] ## using last value of delta, can be changed
    for i in range(delamination.shape[0]-1):
        x = delamination[i, np.nonzero(delamination[i, :])] * r[np.nonzero(delamination[i, :])]
        y = delamination[i, np.nonzero(delamination[i, :])] * z_h[i]
        ax_del1.plot(x[0],y[0], color = 'b')
        ax_del1.plot(-np.flip(x[0]), y[0], color='b')
    i += 1
    x = delamination[i, np.nonzero(delamination[i, :])] * r[np.nonzero(delamination[i, :])]
    y = delamination[i, np.nonzero(delamination[i, :])] * z_h[i]
    ax_del1.plot(-np.flip(x[0]), y[0], color='b')
    ax_del1.plot(x[0], y[0], color='b')
    ax_del1.set_xlabel(r'Distance from hole centre, $r$, [mm]')
    ax_del1.set_ylabel(r'Normalised ply interface location $\frac{z}{h}$ [-]')
    # ax_del1.legend()
    ax_del1.minorticks_on()
    ax_del1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax_del1.set_title(r'Delaminations at ply interfaces, E = %1.2f J' %(E_tot / 1000))
    ax_del1.set_ylim(0,1)

Ez = laminate_class.p.Ey
Er = laminate.ELm
vrt = laminate.vLTm
vrz = 0.0996 ### calculatre
Gzr = laminate_class.p.Gxy
K1 = (1 - imp.v**2) / np.pi / imp.E
alpha = Er / Ez
beta = 1 / (1 - vrt - 2 * vrz ** 2 * alpha)

A11 = Ez * (1 - vrt) * beta
A22 = Er * beta * (1 - vrz ** 2 * alpha) / (1 + vrt)
A12 = Er * vrz * beta

K2_num = sqrt(A22) * sqrt((sqrt(A11 * A22) + Gzr)**2 - (A12 + Gzr)**2)
K2_den = 2 * np.pi * sqrt(Gzr) * (A11 * A22 - A12**2)
K2 = K2_num / K2_den

k_ind = 4 * sqrt(imp.R) / (3 * np.pi * (K1 + K2))