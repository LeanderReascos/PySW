import sympy as sp
import numpy as np
from Modules.classes import Operator, Coefficient, Term, Expression
from Modules.solver import *
#%%
hbar = Coefficient('hbar', order=0)
omega = Coefficient('omega', order=0)
Omega_z = Coefficient('Omega_z', order=0)
g = Coefficient('g', order=1)

a = Operator('a', subspace='oscillator', is_infinite=True, mat_representation=sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
a_dag = Operator('{a^\\dagger}', subspace='oscillator', is_infinite=True, mat_representation=sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

a.add_commutation_relation(a_dag, 1)
a_dag.add_commutation_relation(a, -1)

X = Operator('sigma_x', mat_representation=sp.Matrix([[0, 1], [1, 0]]), subspace='spin')
Z = Operator('sigma_z', mat_representation=sp.Matrix([[1, 0], [0, -1]]), subspace='spin')
Y = Operator('tau_y', mat_representation=sp.Matrix([[0, -1j], [1j, 0]]), subspace='charge')

H = hbar * omega * a_dag * a + hbar * Omega_z * 0.5 * Z + hbar * g * (a + a_dag) * X

#%%

def solver(H, order):
    H_expanded = H.domain_expansion()[0]
    H_0 = H_expanded.group_by_order()[0]
    H_final = deepcopy(H_0)
    Bk = 0

    S = 0

    for k in range(1, order+1):
        print(f"Order {k}")
        H_below_k = sum(H_expanded.group_by_order().get(j, 0) for j in range(k+1))
        Vk = H_below_k.group_by_diagonal()[False].group_by_order().get(k, 0)
        S_k, symbols_sk = get_ansatz(Vk + Bk)
        solution_sk = dict(zip(symbols_sk, [0 for _ in symbols_sk]))
        #display(Bk)
        for key, value in ((H_0 | S_k) + (Vk + Bk)).group_by_infinite_terms().items():
            mats = sp.Add(*[sp.Mul(*map(lambda x: x.representation , term.info["coeff"])) * term.mat_representation["finite"] for term in value.terms])
            solution_sk.update(sp.solve(mats, symbols_sk))
        
        S_k_solved = 0
        for term in S_k.terms:
            term.mat_representation["finite"] = term.mat_representation["finite"].subs(solution_sk)
            if sum(sp.Abs(term.mat_representation["finite"])) == 0:
                continue
            S_k_solved += term
        S += S_k_solved
        display(S)
        tmp_H = float(1 / sp.factorial(k)) * H_below_k.nested_commutator(S, k)
        display(tmp_H)
        H_final += tmp_H.group_by_diagonal().get(True, Expression(Term(Coefficient(0)))).group_by_order().get(k, 0)
        display(H_final)
        Bk = tmp_H.group_by_diagonal()[False].group_by_order()[k+1]
    
    return H_final