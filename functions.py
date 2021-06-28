import numpy as np
from numpy.linalg import inv

def get_A(t, laminas):
    ### assumes constant-thickness layers
    A = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Aij = 0
            for lamina in laminas:
                Aij += laminas[lamina][i, j] * t
            A[i, j] = Aij
    return A


def get_D(t, laminas):
    ### assumes constant-thickness layers
    n_layers = max(laminas.keys())
    D = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Dij = 0
            for layer in laminas:
                z_k1 =  - t * n_layers / 2 + (layer - 1) * t
                z_k = - t * (n_layers / 2 - 1) + (layer - 1) * t
                Dij += laminas[layer][i, j] * (z_k**3 - z_k1**3) / 3
            D[i, j] = Dij
    return D

def get_B(t, laminas):
    ### assumes constant-thickness layers
    n_layers = max(laminas.keys())
    B = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Bij = 0
            for layer in laminas:
                z_k1 =  - t * n_layers / 2 + (layer - 1) * t
                z_k = - t * (n_layers / 2 - 1) + (layer - 1) * t
                Bij += laminas[layer][i, j] * (z_k**2 - z_k1**2) / 2
            B[i, j] = Bij
    return B

def get_ABD(t, laminas):
    A = get_A(t, laminas)
    B = get_B(t, laminas)
    D = get_D(t, laminas)
    ABD = np.zeros((6, 6))
    ABD[:3, :3] = A
    ABD[:3, 3:] = B
    ABD[3:, :3] = B
    ABD[3:, 3:] = D
    return ABD

def offAxis_to_inAxis(quantity, theta, transform = 'stress', angle='radians'):
    '''
    Transform stress or strain state from off-axis to in-axis
    :param quantity: np.array, stress or strain
    :param theta: float, angle (default in radians, specify for degrees)
    :param transform: string, specifies if transform stress or strain. Default is stress
    :param angle: string, specifies if angle unit is rad or deg. Default is radians
    :return: tranformed stress or strain state
    '''

    if angle=='degrees':
        theta = np.deg2rad(theta)
    elif angle == 'radians':
        pass
    else:
        raise TypeError("Keyword argument for 'angle' not understood")
    # print(theta)
    m = np.cos(theta)
    n = np.sin(theta)
    T = np.zeros((6, 6))
    if transform == 'stress':
        T[0, 0] = m**2
        T[0, 1] = n**2
        T[0, 5] = 2 * m * n
        T[1, 0] = n**2
        T[1, 1] = m**2
        T[1, 5] = - 2 * m * n
        T[2, 2] = 1
        T[3, 3] = m
        T[3, 4] = -n
        T[4, 3] = n
        T[4, 4] = m
        T[5, 0] = - m * n
        T[5, 1] = m * n
        T[5, 5] = (m**2 - n**2)
    elif transform == 'strain':
        T[0, 0] = m**2
        T[0, 1] = n**2
        T[0, 5] = m * n
        T[1, 0] = n**2
        T[1, 1] = m**2
        T[1, 5] = - m * n
        T[2, 2] = 1
        T[3, 3] = m
        T[3, 4] = -n
        T[4, 3] = n
        T[4, 4] = m
        T[5, 0] = - 2 * m * n
        T[5, 1] = 2 * m * n
        T[5, 5] = (m**2 - n**2)
    else:
        raise TypeError("Keyword argument for 'transform' not understood")
    quantity_inAxis = T @ quantity
    return quantity_inAxis, T

def C_prime(C, T):
    '''

    :param C: np.array, stiffness matrix
    :param T: np.array, transformation matrix
    :return: np.array, transformed stiffness matrix
    '''
    C_prime = inv(T) @ C @ T
    return C_prime

def get_S_orthotropic(E1, E2, E3, v12, v13, v23, G12, G13, G23):
    S = np.zeros((6, 6))
    S[0, 0] = 1 / E1
    S[0, 1] = - v12 / E2
    S[0, 2] = - v13 / E3
    S[1, 0] = - v12 / E1
    S[1, 1] = 1 / E2
    S[1, 2] = - v23 / E3
    S[2, 0] = - v13 / E1
    S[2, 1] = - v23 / E2
    S[2, 2] = 1 / E3
    S[3, 3] = 1 / G23
    S[4, 4] = 1 / G13
    S[5, 5] = 1 / G12
    return S
def get_S_UD(E1, E2, G12, v12, v23):
    S = np.zeros((6, 6))
    S[0, 0] = 1 / E1
    S[0, 1] = - v12 / E2
    S[0, 2] = - v12 / E2
    S[1, 0] = - v12 / E1
    S[1, 1] = 1 / E2
    S[1, 2] = - v23 / E2
    S[2, 0] = - v12 / E1
    S[2, 1] = - v23 / E2
    S[2, 2] = 1 / E2
    S[3, 3] = 2 * (1 + v23) / E2
    S[4, 4] = 1 / G12
    S[5, 5] = 1 / G12
    return S

def get_C_orthotropic(E1, E2, E3, v12, v13, v23, G12, G13, G23):
    C = np.zeros((6, 6))
    Delta = (1 - v12 * v12 - v23 * v23 - v13 * v13 - 2 * v12 * v13 * v23) / (E1 * E2 * E3)
    C[0, 0] = (1 - v23 * v23) / (E2 * E3 * Delta)
    C[0, 1] = (v12 + v23 * v13) / (E1 * E3 * Delta)
    C[0, 2] = (v13 + v12 * v23) / (E1 * E2 * Delta)
    C[1, 0] = C[0, 1] #not indicated in the slides but logical
    C[1, 1] = (1 - v13 * v13) / (E1 * E3 * Delta)
    C[1, 2] = (v23 * v12 * v13) / (E1 * E2 * Delta)
    C[2, 2] = (1 - v12 * v12) / (E1 * E2 * Delta)
    C[3, 3] = G23
    C[4, 4] = G13
    C[5, 5] = G12
    return C

def get_Q(S):
    Q = np.zeros((3, 3))
    D = S[0, 0] * S[1, 1] - S[0, 1]**2
    Q[0, 0] = S[1, 1] / D
    Q[1, 1] = S[0, 0] / D
    Q[0, 1] = - S[0, 1] / 2
    Q[1, 0] = - S[0, 1] / 2
    Q[2, 2] = 1 / S[5, 5]
    return Q

def lamina(Qxx, Qyy, Qxy, GLT, theta):
    Q = np.zeros((3,3))
    m = np.cos(theta)
    n = np.sin(theta)
    Q[0, 0] = m**4 * Qxx + n**4 * Qyy + 2 * m**2 * n**2 * Qxy + 4 * m**2 * n**2 * GLT
    Q[1, 1] = n**4 * Qxx + m**4 * Qyy + 2 * m**2 * n**2 * Qxy + 4 * m**2 * n**2 * GLT
    Q[2, 2] = m**2 * n**2 * Qxx + m**2 * n**2 * Qyy - 2 * m**2 * n**2 * Qxy + (m**2 - n**2)**2 * GLT
    Q[0, 1] = m**2 * n**2 * Qxx + m**2 * n**2 * Qyy + (m**4 + n**4) * Qxy - 4 * m**2 * n**2 * GLT
    Q[1, 0] = Q[0, 1]
    Q[0, 2] = m**3 * n * Qxx - m * n**3 * Qyy + (m * n**3 - m**3 * n) * Qxy + 2 * (m * n**3 - m**3 * n) * GLT
    Q[1, 2] = m * n**3 * Qxx - m**3 * n * Qyy + (m**3 * n - m * n**3) * Qxy + 2 * (m**3 * n - m * n**3) * GLT
    Q[2, 0] = Q[0, 2]
    Q[2, 1] = Q[1, 2]
    return Q

def get_A(t, laminas):
    ### assumes constant-thickness layers
    A = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Aij = 0
            for lamina in laminas:
                Aij += laminas[lamina][i, j] * t
            A[i, j] = Aij
    return A

def get_D(t, laminas):
    ### assumes constant-thickness layers
    n_layers = max(laminas.keys())
    D = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Dij = 0
            for layer in laminas:
                z_k1 =  - t * n_layers / 2 + (layer - 1) * t
                z_k = - t * (n_layers / 2 - 1) + (layer - 1) * t
                Dij += laminas[layer][i, j] * (z_k**3 - z_k1**3) / 3
            D[i, j] = Dij
    return D

def get_B(t, laminas):
    ### assumes constant-thickness layers
    n_layers = max(laminas.keys())
    B = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Bij = 0
            for layer in laminas:
                z_k1 =  - t * n_layers / 2 + (layer - 1) * t
                z_k = - t * (n_layers / 2 - 1) + (layer - 1) * t
                Bij += laminas[layer][i, j] * (z_k**2 - z_k1**2) / 2
            B[i, j] = Bij
    return B

def get_ABD(t, laminas):
    A = get_A(t, laminas)
    B = get_B(t, laminas)
    D = get_D(t, laminas)
    ABD = np.zeros((6, 6))
    ABD[:3, :3] = A
    ABD[:3, 3:] = B
    ABD[3:, :3] = B
    ABD[3:, 3:] = D
    h = max(laminas.keys()) * t
    return ABD, h

def Transform_2D(quantity, theta, transform = 'stress', angle='radians', initial = 'off-axis'):
    '''
    Transform stress or strain state from off-axis to in-axis
    :param quantity: np.array, stress or strain
    :param theta: float, angle (default in radians, specify for degrees)
    :param transform: string, specifies if transform stress or strain. Default is stress
    :param angle: string, specifies if angle unit is rad or deg. Default is radians
    :return: tranformed stress or strain state
    '''

    if angle=='degrees':
        theta = np.deg2rad(theta)
    elif angle == 'radians':
        pass
    else:
        raise TypeError("Keyword argument for 'angle' not understood")

    m = np.cos(theta)
    n = np.sin(theta)
    T = np.zeros((3, 3))
    if transform == 'strain':
        T[0, 0] = m**2
        T[0, 1] = n**2
        T[0, 2] = m * n
        T[1, 0] = n**2
        T[1, 1] = m**2
        T[1, 2] = - m * n
        T[2, 0] = - 2 * m * n
        T[2, 1] = 2 * m * n
        T[2, 2] = m**2 - n**2
    elif transform == 'stress':
        T[0, 0] = m**2
        T[0, 1] = n**2
        T[0, 2] = 2 * m * n
        T[1, 0] = n**2
        T[1, 1] = m**2
        T[1, 2] = - 2 * m * n
        T[2, 0] = - m * n
        T[2, 1] = m * n
        T[2, 2] = m**2 - n**2
    else:
        raise TypeError("Keyword argument for 'transform' not understood")
    # quantity_inAxis = T @ quantity
    # quantity_inAxis = np.linalg.solve(T, quantity)
    if initial == 'off-axis':
        quantity_inAxis = T @ quantity
    elif initial == 'in-axis':
        quantity_inAxis = np.linalg.solve(T, quantity)
    else:
        raise TypeError("Keyword argument for 'initial' not understood")
    return quantity_inAxis, T

def solve_quadratic(a, b, c):
    delta = b**2 - 4 * a * c
    solve1 = (- b + np.sqrt(delta)) / 2 / a
    solve2 = (- b - np.sqrt(delta)) / 2 / a
    return solve1, solve2
