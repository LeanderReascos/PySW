from Modules.classes import *
from Modules.utils import S

def get_ansatz(H):
    H_dim_of_finite_space = H.get_dim_of_finite_space()

    Hofd = H.group_by_diagonal().get(False, 0)
    order_dict = Hofd.group_by_order()
    min_order = min(list(order_dict.keys()))
    ansatz = S(H_dim_of_finite_space, 'S^{(0)}')

    if H.is_infinite:
        V_k = order_dict[min_order]
        
        infinite_expression = V_k.group_by_infinite().get(True, 0)
        inifnite_operators_dict = {}

        for term in infinite_expression.terms:
            term_inifinite = term.get_infinite()
            inifnite_operators_dict.update({str(term_inifinite) : term_inifinite})
        ansatz += sum([S(H_dim_of_finite_space, f'S^{{({i+1})}}') * inifnite_operator for i, (_, inifnite_operator) in enumerate(inifnite_operators_dict.items())])
        
        return ansatz
    
    return ansatz

def solver(H, order):

    

    ansatz = get_ansatz(H)
    return ansatz

