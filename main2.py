import main1
import numpy as np
from numpy import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
import seaborn as sns
import laminate_class
import impactor as imp
from matplotlib import cm
sns.set()

plot_delam = True
plot_knockdown = True
plt.close('all')
### problems (e) and (f)

# delamination = main1.delamination
k_dels = np.empty(0)
E_dels = np.empty(0)
k_pristine = np.empty(0)
E_pristine = np.empty(0)
delta_s_max = np.empty(0)
delta_pen_max = main1.delta_s_max.max()
E_norms = np.empty(0)
E_tots = np.empty(0)
for key in main1.delamins_dict:
    print('%1.2f %% completed' % (key / delta_pen_max * 100))
    delamination = main1.delamins_dict[key]
    if np.sum(delamination) > 0:
        delamination = np.hstack([np.flip(delamination), delamination])
        vertical_lines = np.zeros(delamination.shape[1])

        for i in range(delamination.shape[0]):
            for j in range(1, delamination.shape[1]):
                check = delamination[i, j]
                check_before = delamination[i, j - 1]
                if check != check_before and check:
                    vertical_lines[j] = 1
                elif check != check_before and not check:
                    vertical_lines[j-1] = 1

        r = main1.r
        r_sym = np.hstack([-np.flip(r), r])
        z_h = main1.z_h

        ### scan horizontally until we see a vertical line. Once we see a vertical line we start to scan vertically to find delaminated zones.
        ## we need to identify delaminated zones. Once we have it, then we gotta find the array that will give us the stacking sequence
        ### use laminate class and stacking sequence to obtain the stiffness. Multiply stiffness by length to find

        ### get column indices of vertical lines

        r_vertical = np.empty(0)
        for j in range(vertical_lines.shape[0]):
            if vertical_lines[j]:
                r_vertical = np.append(r_vertical, r_sym[j])
        delta_r = np.diff(r_vertical)

        ks = np.zeros(delta_r.shape[0] // 2 + 1)

        num_del = 0
        vertical_idx = np.nonzero(vertical_lines)[0]
        vertical_idx_diff = np.diff(vertical_idx)
        check = np.any(vertical_idx_diff == 1)
        assert check == False, 'Decrease delta_x in main1 (line 10)'
        zones = 0
        for j in range(vertical_lines.shape[0] // 2):
            check_prev = 0
            check = vertical_lines[j]
            if check:
                n_check = 0
                if np.sum(delamination[:, j + 1]) != 0:
                    for i in range(delamination.shape[0] // 2 +1):
                        check2 = delamination[i, j+1]
                        if check2:
                            n_check += 1
                            stack = main1.stack[check_prev:i]
                            # print('================================================================')
                            # print(delamination[:, j + 1], r_sym[j])
                            # print(stack)
                            check_prev = i
                            laminate = laminate_class.laminate(stack_init=stack)
                            laminate_sym = laminate_class.laminate(stack_init=np.flip(stack))
                            ks[zones] += laminate.ETm * laminate.h / delta_r[zones] + laminate_sym.ETm * laminate_sym.h / delta_r[zones]
                    # print(ks[zones] * delta_r[zones], zones, 'if')




                else:
                    stack = main1.stack
                    laminate = laminate_class.laminate(stack_init=stack)
                    ks[zones] += laminate.ETm * laminate.h / delta_r[zones]
                    # print(ks[zones], zones, 'else')

                zones += 1

        ks = np.hstack([ks, np.flip(ks[:-1])])

        k_inv = 0
        for k in ks:
            k_inv += 1 / k

        k_final = 1 / k_inv
        k_original = main1.laminate.ETm * main1.laminate.h / (r_sym[vertical_idx[-1]] - r_sym[vertical_idx[0]])

        ### equivalent moduli
        E_original = main1.laminate.ETm
        E_final = k_final * (r_sym[vertical_idx[-1]] - r_sym[vertical_idx[0]]) / main1.laminate.h

        k_dels = np.append(k_dels, k_final * (r_sym[vertical_idx[-1]] - r_sym[vertical_idx[0]]))
        k_pristine = np.append(k_pristine, k_original * (r_sym[vertical_idx[-1]] - r_sym[vertical_idx[0]]))
        E_dels = np.append(E_dels, E_final)
        E_pristine = np.append(E_pristine, E_original)
        delta_s_max = np.append(delta_s_max, key)
        E_norms = np.append(E_norms, main1.E_norms[key])
        E_tots = np.append(E_tots, main1.E_tots[key])
    else:
        k_original = main1.laminate.ETm * main1.laminate.h
        k_final = k_original
        ### equivalent moduli
        E_original = main1.laminate.ETm
        E_final = E_original

        k_dels = np.append(k_dels, k_final)
        k_pristine = np.append(k_pristine, k_original)
        E_dels = np.append(E_dels, E_final)
        E_pristine = np.append(E_pristine, E_original)
        delta_s_max = np.append(delta_s_max, key)
        E_norms = np.append(E_norms, main1.E_norms[key])
        E_tots = np.append(E_tots, main1.E_tots[key])
        print('delta_max = %1.4f mm' %key, 'skipped')
        continue


# print('==========================================================')
# print('Stiffness per unit width')
# print('k w/o delaminations: %1.3f kN / mm^2' % (k_original / 1000))
# print('k  w  delaminations: %1.3f kN / mm^2' % (k_final / 1000))
#
# print('==========================================================')
#
# print('Equivalent Elastic Moduli')
# print('E w/o delaminations: %1.2f GPa' % (E_original / 1000))
# print('E  w  delaminations: %1.2f GPa' % (E_final / 1000))

if plot_knockdown:
    fig_delta, ax_delta = plt.subplots()

    ax_delta.plot(delta_s_max, k_dels / k_pristine)
    ax_delta.set_xlabel(r'$\delta_\mathrm{max}$ [mm]')
    ax_delta.set_ylabel(r'$\frac{k_\mathrm{del}}{k_\mathrm{pristine}}$ [-]')
    ax_delta.minorticks_on()
    ax_delta.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax_delta.set_title(r'Delaminated-to-pristine stiffness ratio vs indentation')

    fig_energy, ax_energy = plt.subplots()

    ax_energy.plot(E_norms, k_dels / k_pristine)
    ax_energy.set_xlabel(r'Normalised Impact Energy [lb-ft / in]')
    ax_energy.set_ylabel(r'$\frac{k_\mathrm{del}}{k_\mathrm{pristine}}$ [-]')
    ax_energy.minorticks_on()
    ax_energy.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax_energy.set_title(r'Delaminated-to-pristine stiffness ratio vs normalised impact energy')

    fig_en, ax_en = plt.subplots()
    ax_en.plot(E_tots / 1000, k_dels / k_pristine)
    ax_en.set_xlabel(r'Impact Energy [J]')
    ax_en.set_ylabel(r'$\frac{k_\mathrm{del}}{k_\mathrm{pristine}}$ [-]')
    ax_en.minorticks_on()
    ax_en.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax_en.set_title(r'Delaminated-to-pristine stiffness ratio vs impact energy')


key_index = 6
delamination_plot = main1.delamins_dict[delta_s_max[key_index]]
vertical_lines = np.zeros(delamination_plot.shape[1])

for i in range(delamination_plot.shape[0]):
    for j in range(1, delamination_plot.shape[1]):
        check = delamination_plot[i, j]
        check_before = delamination_plot[i, j - 1]
        if check != check_before and check:
            vertical_lines[j] = 1
        elif check != check_before and not check:
            vertical_lines[j - 1] = 1
E_tot = E_norms[key_index]

if plot_delam:
    fig_del1, ax_del1 = plt.subplots()
    for i in range(delamination.shape[0] - 1):
        x = delamination_plot[i, np.nonzero(delamination_plot[i, :])] * r[np.nonzero(delamination_plot[i, :])]
        y = delamination_plot[i, np.nonzero(delamination_plot[i, :])] * z_h[i]
        ax_del1.plot(x[0], y[0], color='b')
        ax_del1.plot(-np.flip(x[0]), y[0], color='b')
    i += 1
    x = delamination_plot[i, np.nonzero(delamination_plot[i, :])] * r[np.nonzero(delamination_plot[i, :])]
    y = delamination_plot[i, np.nonzero(delamination_plot[i, :])] * z_h[i]
    ax_del1.plot(-np.flip(x[0]), y[0], color='b')
    ax_del1.plot(x[0], y[0], color='b')

    for i in range(vertical_lines.shape[0]):
        if vertical_lines[i]:
            ax_del1.plot(np.ones(2) * r_sym[i], np.linspace(0,1,2), color = 'r')

    ax_del1.set_xlabel(r'Distance from hole centre, $r$, [mm]')
    ax_del1.set_ylabel(r'Normalised ply interface location $\frac{z}{h}$ [-]')
    # ax_del1.legend()
    ax_del1.minorticks_on()
    ax_del1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax_del1.set_title(r'Delaminations at ply interfaces, $E_\mathrm{norm}$ = %1.2f lb-ft / in' % (E_tot))
    ax_del1.set_ylim(0, 1)