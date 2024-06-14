import textwrap
import warnings
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


def RD_solve(expr, unknowns):
    eqs_dict = group_by_finite_operators(expr)
    equations_to_solve = list(eqs_dict.values())
    sols = solve(equations_to_solve, unknowns)
    return sols


def solver(H, composite_basis, order=2, full_diagonal=True, commutation_relations=None):

    from_list_to_key_lenght = lambda l: ('-'.join([str(i) for i in l]), len(l))

    terms = H.expand().as_ordered_terms()
    subsaces = []
    for term in terms:
        if not term.has(RDBoson):
            
            continue
        for factor in term.as_ordered_factors():
            if factor.has(RDBoson):
                subsaces.append(factor.subspace)

    boson_subspaces = list(set(subsaces))

    list_keys_full = generate_keys(order)
    keys = rearrange_keys(list_keys_full)
    keys.pop()

    rational_factorial = [Rational(1, factorial(i)) for i in range(order + 1)]

    H_ordered = group_by_order(H)
    elementes_ordered = {str(key): value for key, value in H_ordered.items()}

    H0 = elementes_ordered.get('0', 0)
   
    Vk_dict = {}
    Bk_dict = {}
    H_final = 0
    
    for key, value in H_ordered.items():
        Hk = group_by_diagonal(value)
        H_final += Hk.get(True, 0)
        Vk_dict[key] = Hk.get(False, 0)
          
    S = {}
        
    for key in keys:
        k_total, l_total = from_list_to_key_lenght(key)
        order_it = np_sum(key)
        k_last, _ = from_list_to_key_lenght(key[:-1])
        k, _ = from_list_to_key_lenght([key[-1]])
                
        if l_total == 1:
                        continue
            
        if l_total == 2 and key[0] == 0:
            Vk = Vk_dict.get(order_it, 0)
            Bk = Bk_dict.get(order_it, 0)

            
            Vk_plus_Bk = Vk + Bk
            if Vk_plus_Bk == 0:
                S[k] = 0
                continue

            Sk, symbols_s = get_ansatz(Vk_plus_Bk, composite_basis)
            
            S_k_grouped = group_by_infinite_operators(Sk, commutation_relations)
            S_k_solved = 0

            elementes_ordered[k_total] = - Vk_plus_Bk
            eq = apply_commutation_relations(expand_commutator(Commutator(H0, Sk) + Vk_plus_Bk).doit(), commutation_relations)
            
            expression_to_solve = composite_basis.project(eq).simplify().expand()
            sols = {s : 0 for s in symbols_s}

            group_by_infinite_operators_dict = group_by_infinite_operators(expression_to_solve, commutation_relations)

            raise_warning(group_by_infinite_operators_dict, boson_subspaces)

            for key, value in group_by_infinite_operators_dict.items():
                if value == 0:
                    continue
                solution_dict = RD_solve(value, symbols_s)
                sols.update(solution_dict)
                sk = S_k_grouped.get(key, 0)
                if sk == 0:
                    continue
                S_k_solved += key * sk.subs(sols)
            
            S[k] = S_k_solved
            continue
        
        prev_term = elementes_ordered.get(k_last, 0)
        Sk = S[k]
        new_term =  composite_basis.project(expand_commutator(Commutator(prev_term, Sk)).doit()).simplify().expand()
        
        
        elementes_ordered[k_total] = new_term

        new_term_to_the_hamiltonian = (rational_factorial[l_total - 1] *new_term).expand()

        if not full_diagonal:
            H_final += new_term_to_the_hamiltonian
            continue
        Hk_new = group_by_diagonal(new_term_to_the_hamiltonian)
        Bk_dict[order_it] = Bk_dict.get(order_it, 0) + Hk_new.get(False, 0)
        H_final += Hk_new.get(True, 0)
    
    if commutation_relations:
        H_final = apply_commutation_relations(H_final, commutation_relations)

    return H_final, S

def raise_warning(group_by_infinite_operators_dict, boson_subspaces):
    # Raise warning if inconsistency in equations to solve
    eqn_keys = list(group_by_infinite_operators_dict.keys())
    counted_bosons = list(map(count_bosons, eqn_keys))

    unique_bosons = {subspace: [] for subspace in boson_subspaces}
    

    for term in counted_bosons:
        if term is None:
            continue
        res_count_dict = {subspace : 0 for subspace in boson_subspaces}
        for subspace, boson_count in term.items():
            res_count = boson_count.get("creation", 0) - boson_count.get("annihilation", 0)
            # res_count > 0 means there are more creation operators than annihilation operators
            # res_count < 0 means there are more annihilation operators than creation operators
            res_count_dict[subspace] = res_count
        for subspace, count in res_count_dict.items():
            unique_bosons[subspace].append(count)

    unique_bosons_count = array(list(unique_bosons.values())).T
    check_unique_bosons = set(list(map(lambda x: ','.join(map(str, x)), unique_bosons_count)))
    

    if len(check_unique_bosons) != len(unique_bosons_count):
        warnig_message = """Inconsistency in equations to solve. The equations used to solve for S are not linearly independent. Full diagonalization up to the selected order for the given problem is not yet supported. We recommend to either lower perturbation order, or otherwise use the truncated solver.
        """

        warnig_message =  textwrap.fill(warnig_message, width=100)

        warnings.warn('\n' + warnig_message)
            