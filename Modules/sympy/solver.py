
from sympy import solve, factorial
from Modules.sympy.classes import *
from Modules.sympy.utils import *

from itertools import product

def get_ansatz(Vk, composite_basis):

    to_separate = list(group_by_infinite_operators(Vk).keys())
    order = min(list(group_by_order(Vk).keys()))
    ansatz = 0
    symbol_s = []
    for idx, term in enumerate(to_separate):
        tmp_s = [RDsymbol(f"S^{idx}_{i}", order=order) for i in range(composite_basis.dim**2)]
        symbol_s += tmp_s
        ansatz += term * (tmp_s * composite_basis._basis).sum()
    return ansatz, symbol_s

def RD_solve(expr, unknowns):
    eqs_dict = group_by_finite_operators(expr)
    equations_to_solve = list(eqs_dict.values())
    sols = solve(equations_to_solve, unknowns)
    return sols


def solver(H, Composite_basis, order = 2):

    H_ordered = group_by_order(H)
    H0 = H_ordered.get(0, 0)
    Bk = 0

    S = 0
    H_final = H0
    H_block_diagonal = 0

    for k in range(1, order+1):
        print()
        print("-"*50)
        print(f"Order {k}")
        H_below_k = sum(H_ordered.get(j, 0) for j in range(k+1))
        Vk = H_ordered.get(k, 0)
        if Vk == 0 and Bk == 0:
            print("Skipping")
            continue
        Sk, symbol_s = get_ansatz(Vk + Bk, Composite_basis)

        print("Got ansatz")

        S_k_grouped = group_by_infinite_operators(Sk)
        S_k_solved = 0
        
        expression_to_solve = Composite_basis.project(expand_commutator(Commutator(H0, Sk) + Vk + Bk).doit()).simplify().expand()
        sols = {s : 0 for s in symbol_s}

        print("Solving")

        group_by_infinite_operators_dict = group_by_infinite_operators(expression_to_solve)

        print("\tGrouped by infinite operators")

        for key, value in group_by_infinite_operators_dict.items():
            print(f"\t\tKey: {key}")
            if value == 0:
                continue
            solution_dict = RD_solve(value, symbol_s)
            print("\t\tGot Solutions")
            sols.update(solution_dict)
            sk = S_k_grouped.get(key, 0)
            if sk == 0:
                continue
            S_k_solved += key * sk.subs(sols)

        print("Got S_k_solved")

        S += S_k_solved
        tmp_H = Composite_basis.project(expand_commutator(Commutator(H0, S_k_solved)).doit()).simplify().expand()
        tmp_H += (1 / factorial(k) * Composite_basis.project(expand_commutator(nested_commutator(H_below_k, S, k)).doit())).simplify().expand()
        print("Got tmp_H")

        temp_H_grouped = group_by_diagonal(tmp_H)
        H_final += group_by_order(temp_H_grouped[True]).get(k, 0) +  group_by_order(temp_H_grouped[True]).get(k+1, 0)
        print("Got H_final")


        Bk = group_by_order(temp_H_grouped[False]).get(k+1, 0)

        H_block_diagonal += Bk
        print("Got Bk")
        print("-"*50)
        print()

    return S, H_final, H_block_diagonal + H_final