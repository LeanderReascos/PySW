from sympy import solve, factorial, symbols, Rational, diag, factorial, zeros, Matrix
from Modules.sympy.classes import *
from Modules.sympy.utils import *

from itertools import product


def generate_keys(order):
    keys = [] # will contain keys
    
    def deapth_first_search(current_key, current_sum):
        if current_sum > order: # if current_sum is bigger than order
            return # return nothing
        if current_key: # if current key is not empty
            keys.append(current_key[:]) # append current key to keys
        start = 0 if not current_key else 1 # if current key is empty, start the next number from 0, else from 1
        for i in range(start, order + 1): # looping over values between start and order
            current_key.append(i) # append current number to the key
            deapth_first_search(current_key, current_sum + i) # recurse down the tree
            current_key.pop() # backtrack

            
            
    deapth_first_search([], 0) # start search with inital sum to zero
    return keys

def custom_sort_key(sublist):
    """
    Custom sorting key function:
    - First, sorts by the sum of sublist elements in increasing order.
    - Second, sorts sublists with equal sums by moving sublists of length 2 containing a 0 in the first element to the bottom.
    """
    sublist_sum = sum(sublist)
    sublist_length = len(sublist)

    if sublist_length == 2 and sublist[0] == 0:
        # Ensure sublists of length 2 with a 0 in the first element go to the bottom
        return (sublist_sum, 1)
    else:
        return (sublist_sum, 0)

def rearrange_keys(list_of_sublists):
    # Sort sublists based on custom key
    sorted_sublists = sorted(list_of_sublists, key=custom_sort_key)

    return sorted_sublists


def get_ansatz(dim):
    symbols_s = [symbols(f"c_{{{i}}}") for i in range(dim**2)]
    ansatz = zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            ansatz[i, j] = symbols_s[i * dim + j]
    return ansatz, symbols_s


def solver(H, list_subspaces, order=2, full_diagonal=True):

    from_list_to_key_lenght = lambda l: ('-'.join([str(i) for i in l]), len(l))

    list_keys_full = generate_keys(order)
    keys = rearrange_keys(list_keys_full)
    keys.pop()

    rational_factorial = [Rational(1, factorial(i)) for i in range(order + 1)]

    for i, key in enumerate(keys):
        k_str, l = from_list_to_key_lenght(key)

    H_ordered = group_by_order(H)
    elementes_ordered = {str(key): get_matrix(value, list_subspaces) for key, value in H_ordered.items()}

    H0 = elementes_ordered.get('0', 0)
    dim = H0.shape[0]
    zero_matrix = zeros(dim)

    Vk_dict = {}
    Bk_dict = {}
    H_final = zeros(H0.shape[0])
    
    for key, value in H_ordered.items():
        Hk = group_by_diagonal(value)
        H_final += get_matrix(Hk.get(True, 0), list_subspaces)
        Vk_dict[key] = get_matrix(Hk.get(False, 0), list_subspaces)


    S = {}

    for key in keys:
        k_total, l_total = from_list_to_key_lenght(key)
        order_it = np_sum(key)
        k_last, _ = from_list_to_key_lenght(key[:-1])
        k, _ = from_list_to_key_lenght([key[-1]])

        if l_total == 1:
            continue
            
        if l_total == 2 and key[0] == 0:
            Vk = Vk_dict.get(order_it, zero_matrix)
            Bk = Bk_dict.get(order_it, zero_matrix)

            print(f"Solving S_{k}")

            Vk_plus_Bk = Vk + Bk

            if Vk_plus_Bk == 0:
                S[k] = 0
                continue

            Sk, symbols_s = get_ansatz(dim)

            elementes_ordered[k_total] = - Vk_plus_Bk
            expression_to_solve = (H0 * Sk - Sk * H0) + Vk_plus_Bk

            sols = {s : 0 for s in symbols_s}
            solution = solve(expression_to_solve, symbols_s, dict=True)[0]
            sols.update(solution)

            S[k] = Sk.subs(sols)
            continue
        
        prev_term = elementes_ordered.get(k_last, zero_matrix)
        Sk = S[k]
        new_term =  (prev_term * Sk - Sk * prev_term).doit()

        elementes_ordered[k_total] = new_term

        new_term_to_the_hamiltonian = rational_factorial[l_total - 1] * new_term

        if not full_diagonal:
            H_final += new_term_to_the_hamiltonian
            continue
        
        Hk_diag = diag(*new_term_to_the_hamiltonian.diagonal())

        Bk_dict[order_it] = Bk_dict.get(order_it, zero_matrix) + new_term_to_the_hamiltonian - Hk_diag
        H_final += Hk_diag

    return H_final, S