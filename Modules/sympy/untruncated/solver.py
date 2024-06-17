import textwrap
import warnings
from sympy import solve, factorial
from Modules.sympy.classes import *
from Modules.sympy.utils import *
from tqdm import tqdm
from itertools import product


def get_ansatz(Vk, composite_basis):
    """
    Generate an ansatz for the given operator and basis.

    This function creates an ansatz by separating terms with infinite operators and
    combining them with the composite basis using symbolic coefficients.

    Parameters:
    -----------
    Vk : sympy.Expr
        The operator to be separated and combined with the composite basis.
    composite_basis : CompositeBasis
        The composite basis to be used in the ansatz.

    Returns:
    --------
    tuple
        - ansatz (sympy.Expr): The generated ansatz.
        - symbol_s (list): A list of symbolic coefficients used in the ansatz.

    Examples:
    ---------
    >>> Vk = ...  # some operator
    >>> composite_basis = ...  # some composite basis
    >>> ansatz, symbols = get_ansatz(Vk, composite_basis)
    """
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
    """
    Generate all possible keys with sums of their elements up to a given order.

    This function generates all combinations of keys where the sum of the elements 
    does not exceed the specified order. The keys are generated using a depth-first 
    search algorithm.

    Parameters:
    -----------
    order : int
        The maximum sum for the keys.

    Returns:
    --------
    List[List[int]]
        A list of lists where each inner list is a key whose elements sum up to 
        the given order.

    Examples:
    ---------
    >>> generate_keys(2)
    [[], [0], [0, 0], [1], [0, 1], [2], [1, 1]]

    >>> generate_keys(1)
    [[], [0], [0, 0], [1]]
    """
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
    Custom sorting key function for sublists.

    This function provides a custom key for sorting sublists based on specific rules:
    - First, sorts by the sum of sublist elements in increasing order.
    - Second, moves sublists of length 2 with a 0 in the first element to the bottom.

    Parameters:
    -----------
    sublist : List[int]
        The sublist to generate the sorting key for.

    Returns:
    --------
    Tuple[int, int]
        A tuple where the first element is the sum of the sublist and the second element 
        is 1 if the sublist length is 2 and the first element is 0, otherwise 0.

    Examples:
    ---------
    >>> custom_sort_key([0, 1])
    (1, 0)

    >>> custom_sort_key([0, 0])
    (0, 0)

    >>> custom_sort_key([1, 2, 3])
    (6, 0)

    >>> custom_sort_key([0, 2])
    (2, 1)
    """
    sublist_sum = sum(sublist)
    sublist_length = len(sublist)

    if sublist_length == 2 and sublist[0] == 0:
        # Ensure sublists of length 2 with a 0 in the first element go to the bottom
        return (sublist_sum, 1)
    else:
        return (sublist_sum, 0)

def rearrange_keys(list_of_sublists):
    """
    Rearrange a list of sublists based on a custom sorting key.

    This function sorts a list of sublists using the custom_sort_key function to 
    determine the order. The sorting is primarily by the sum of the elements in each 
    sublist. Sublists of length 2 with a 0 as the first element are moved to the bottom.

    Parameters:
    -----------
    list_of_sublists : List[List[int]]
        The list of sublists to be sorted.

    Returns:
    --------
    List[List[int]]
        The sorted list of sublists.

    Examples:
    ---------
    >>> rearrange_keys([[0, 1], [0, 0], [1, 2, 3], [0, 2], [1]])
    [[0, 0], [0, 1], [1], [1, 2, 3], [0, 2]]

    >>> rearrange_keys([[2, 3], [0, 3], [1]])
    [[1], [2, 3], [0, 3]]
    """
    # Sort sublists based on custom key
    sorted_sublists = sorted(list_of_sublists, key=custom_sort_key)

    return sorted_sublists


def RD_solve(expr, unknowns):
    """
    Solve a system of equations involving finite operators.

    This function solves a system of equations extracted from the given expression,
    grouping them by finite operators, and then solving for the specified unknowns.

    Parameters:
    -----------
    expr : sympy.Expr
        The expression containing the equations to be solved.
    unknowns : list
        The list of unknowns to solve for.

    Returns:
    --------
    dict
        A dictionary containing the solutions for the unknowns.

    Examples:
    ---------
    >>> expr = ...  # some expression
    >>> unknowns = [...]  # list of unknowns
    >>> solutions = RD_solve(expr, unknowns)
    """
    eqs_dict = group_by_finite_operators(expr)
    equations_to_solve = list(eqs_dict.values())
    sols = solve(equations_to_solve, unknowns)
    return sols


def solver(H, composite_basis, order=2, full_diagonal=True, commutation_relations=None):
    """
    Solve a Hamiltonian using perturbation theory up to a specified order.

    This function solves for the Hamiltonian using a perturbative approach, taking into
    account commutation relations, and generates solutions for each order of perturbation.

    Parameters:
    -----------
    H : sympy.Expr
        The Hamiltonian to be solved.
    composite_basis : CompositeBasis
        The composite basis to be used in the solution.
    order : int, optional
        The order of perturbation theory to use (default is 2).
    full_diagonal : bool, optional
        Whether to fully diagonalize the Hamiltonian (default is True).
    commutation_relations : dict, optional
        The commutation relations to be used (default is None).

    Returns:
    --------
    tuple
        - H_final (sympy.Expr): The final Hamiltonian.
        - S (dict): A dictionary containing the solutions for each order.

    Examples:
    ---------
    >>> H = ...  # some Hamiltonian
    >>> composite_basis = ...  # some composite basis
    >>> H_final, S = solver(H, composite_basis, order=2, full_diagonal=True)
    """

    from_list_to_key_lenght = lambda l: ('-'.join([str(i) for i in l]), len(l))

    terms = H.expand().as_ordered_terms()
    subsaces = []
    for term in terms:
        if not term.has(RDBoson):
            
            continue
        for factor in term.as_ordered_factors():
            if factor.has(RDBoson):
                if isinstance(factor, Pow):
                    factor, _ = factor.as_base_exp()
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
        
    for key in tqdm(keys):
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
    """
    Raise a warning if there is an inconsistency in the equations to solve.

    This function checks for inconsistencies in the equations derived from 
    group_by_infinite_operators. It warns if the equations are not linearly 
    independent, suggesting alternative approaches.

    Parameters:
    -----------
    group_by_infinite_operators_dict : dict
        A dictionary of grouped infinite operators.
    boson_subspaces : list
        A list of boson subspaces.

    Examples:
    ---------
    >>> group_by_infinite_operators_dict = ...  # some dictionary
    >>> boson_subspaces = [...]  # list of boson subspaces
    >>> raise_warning(group_by_infinite_operators_dict, boson_subspaces)
    """
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
            
