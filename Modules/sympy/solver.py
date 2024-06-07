
from sympy import solve, factorial
from Modules.sympy.classes import *
from Modules.sympy.utils import *

def get_ansatz(Vk):
    '''
    Returns ansatz operator for given operator Vk.
    '''
    H_dim_of_finite_space = Vk.dim

    Hofd = H.group_by_diagonal().get(False, 0)
    order_dict = Hofd.group_by_order()
    min_order = min(list(order_dict.keys()))
    ansatz, symbols = S(H_dim_of_finite_space, f'S^{{({0})}}_{min_order}')

    if H.is_infinite:
        V_k = order_dict[min_order]
        
        infinite_expression = V_k.group_by_infinite().get(True, 0)
        inifnite_operators_dict = {}

        for term in infinite_expression.terms:
            term_inifinite = term.get_infinite()
            inifnite_operators_dict.update({str(term_inifinite) : term_inifinite})
        temp = [S(H_dim_of_finite_space, f'S^{{({i+1})}}_{min_order}', order=min_order) for i in range(len(inifnite_operators_dict))]
        Ss = []
        symbols_list = []
        for S_op, symbols_list_ in temp:
            Ss.append(S_op)
            symbols_list += symbols_list_
        ansatz += sum([Ss[i] * inifnite_operator for i, (_, inifnite_operator) in enumerate(inifnite_operators_dict.items())])
        
        return ansatz, symbols + symbols_list
    
    return ansatz, symbols


def solver(H, order):
    """
    H: Hamiltonian Expression
    order: Order of the transformation
    """

    H_ordered = group_by_order(H)
    H0 = H_ordered.get(0, 0)
    Bk = 0

    S = 0
    H_final = H0

    for k in range(1, order+1):
        H_below_k = sum(H_ordered.get(j, 0) for j in range(k+1))
        Vk = group_by_diagonal(H_below_k)[False]
        S_k, symbols_sk = get_ansatz(Vk + Bk)

        S_k_grouped = group_by_infinite_operators(S_k)

        S_k_solved = 0

        expression_to_solve = expand_commutator(Commutator(H0, S_k) + Vk + Bk).doit()
        for key, value in group_by_infinite_operators(expression_to_solve).items():
            value_expanded = domain_expansion(value)
            solution_sk = solve(value_expanded, symbols_sk)
        
            S_k_solved += key * RDOperator(S_k_grouped[key].name, "S", S_k_grouped[key].dim, S_k_grouped[key].matrix.subs(solution_sk)) 

        S += S_k_solved
        tmp_H = 1 / factorial(k) * expand_commutator(nested_commutator(H_below_k, S, k)).doit()

        temp_H_grouped = group_by_diagonal(tmp_H)
        H_final += temp_H_grouped[True]
        Bk = group_by_order(temp_H_grouped[False]).get(k+1, 0)

    return H_final