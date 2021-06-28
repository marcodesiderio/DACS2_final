import functions as f
import numpy as np
import lamina_props as p
import hashin_funcs as h
class laminate:
    __slots__ = ['A', 'B', 'D', 'ABD', 'tply', 'a', 'd', 'stack', 'h', 'vxy', 'vyx',
                 'ELm', 'ETm', 'GLTm', 'vLTm', 'vTLm', 'ELb', 'ETb', 'GLTb', 'vLTb', 'vTLb']

    def __init__(self, stack_init):
        self.stack = stack_init
        self.tply = p.t_ply

        laminas = {}
        i = 1
        for lamina in self.stack:
            angle = np.deg2rad(lamina)
            lamina_props = f.lamina(p.Qxx, p.Qyy, p.Qxy, p.Gxy, angle)
            laminas[i] = lamina_props
            i += 1
        self.ABD, self.h = f.get_ABD(self.tply, laminas)
        self.A = self.ABD[:3, :3]
        self.B = self.ABD[:3, 3:]
        self.D = self.ABD[3:, 3:]
        self.a = np.linalg.inv(self.A)
        self.d = np.linalg.inv(self.D)
        # self.vxy = - self.a[0, 1] / self.a[0, 0]
        # self.vyx = - self.a[0, 1] / self.a[1, 1]
        self.ELm, self.ETm, self.GLTm, self.vLTm, self.vTLm, self.ELb, self.ETb, self.GLTb, self.vLTb, self.vTLb = self.get_elastic_props()

    def fractions(self):
        unique, counts = np.unique(self.stack, return_counts=True)
        fraction = counts / np.sum(counts) * 100
        return unique, fraction, counts

    def get_global_strain(self, F):

        e = np.linalg.solve(self.ABD, F)
        return e
    def get_lamina_SS(self, F):

        e_glob = self.get_global_strain(F)
        i = 1
        stresses = {}
        strains = {}
        for lamina in self.stack:
            z = (- len(self.stack) / 2 * self.tply + self.tply / 2 + (i - 1) * self.tply)
            e_glob_lamina = e_glob[:3] + z * e_glob[3:]
            strain_principal, T = f.Transform_2D(e_glob_lamina, lamina, transform='strain', angle='degrees', initial='off-axis')
            stress_principal = p.Q @ strain_principal
            stresses[i] = stress_principal
            strains[i] = strain_principal
            i += 1
        return strains, stresses

    def hashin_fail(self, F):
        i = 1
        check_matrix = np.ones(len(self.stack), dtype=bool)
        check_fibres = np.ones(len(self.stack), dtype=bool)
        properties = np.ones(len(self.stack))
        strains, stresses = self.get_lamina_SS(F)
        for lamina in self.stack:
            check_matrix[i-1] = h.check_matrix(p.Yt, p.Yc, p.S, p.S, stresses[i][2], 0, 0, stresses[i][1], 0)   ### True = no fail; false = Fail
            check_fibres[i-1] = h.check_fibers(p.Xt, p.Xc, p.S, stresses[i][0], stresses[i][2], 0)        ### True = no fail; false = Fail
            i += 1
        return check_matrix, check_fibres
    def get_elastic_props(self):
        h = self.h
        a = self.a
        d = self.d
        ELm = 1 / h / a[0,0]
        ETm = 1 / h / a[1,1]
        GLTm = 1 / h / a[2,2]
        vLTm = - a[0,1] / a[0,0]
        vTLm = - a[0,1] / a[1,1]

        ELb = 12 / h**3 / d[0,0]
        ETb = 12 / h**3 / d[1,1]
        GLTb = 12 / h**3 / d[2,2]
        vLTb = - d[0,1] / d[0,0]
        vTLb = - d[0,1] / d[1,1]
        return ELm, ETm, GLTm, vLTm, vTLm, ELb, ETb, GLTb, vLTb, vTLb





