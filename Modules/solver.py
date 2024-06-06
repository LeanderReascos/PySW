from Modules.classes import *
from Modules.utils import S

def get_ansatz(H):
    '''
    Now we change it to only receive Vk instead of H NOTE: This is a temporary solution!!!!!!!!!!!!!
    '''
    H_dim_of_finite_space = H.get_dim_of_finite_space()

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

    

    ansatz = get_ansatz(H)
    return ansatz

