from sympy import solve, factorial, symbols, Rational, diag, factorial, zeros, Matrix, Abs
from Modules.sympy.classes import *
from Modules.sympy.utils import *
from tqdm import tqdm
from itertools import product
from warnings import warn


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
    [[], [0], [1], [0, 1], [2], [1, 1]]

    >>> generate_keys(1)
    [[], [0], [1]]

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


def get_ansatz(dim):
    """
    Generate an ansatz matrix and a list of symbolic variables.

    This function creates a square matrix of dimension `dim` filled with unique symbolic
    variables and returns both the matrix and the list of symbolic variables.

    Parameters:
    -----------
    dim : int
        The dimension of the ansatz matrix.

    Returns:
    --------
    tuple
        - ansatz : sympy.Matrix
            A `dim x dim` matrix filled with symbolic variables.
        - symbols_s : list of sympy.Symbol
            A list of symbolic variables used in the ansatz matrix.

    Examples:
    ---------
    >>> get_ansatz(2)
    (Matrix([
    [c_{0}, c_{1}],
    [c_{2}, c_{3}]]), [c_{0}, c_{1}, c_{2}, c_{3}])
    """
    symbols_s = [symbols(f"c_{{{i}}}") for i in range(dim**2)]
    ansatz = zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            if i == j: # skipping elements along diagonal
                continue
            ansatz[i, j] = symbols_s[i * dim + j]
    return ansatz, symbols_s


def solver(H, list_subspaces, order=2, full_diagonal=True):
    """
    Solve the Hamiltonian system by generating and using nested commutators.

    This function decomposes the Hamiltonian into subspaces and iteratively constructs
    the final Hamiltonian using nested commutators and symbolic ansatz matrices. The
    process can generate up to a specified order of terms.

    Parameters:
    -----------
    H : sympy.Matrix
        The Hamiltonian matrix.
    list_subspaces : list of lists
        A list of subspaces for the Hamiltonian decomposition.
    order : int, optional
        The order up to which the nested commutators are computed (default is 2).
    full_diagonal : bool, optional
        If True, only diagonal terms are added to the final Hamiltonian. If False, all terms
        are added (default is True).

    Returns:
    --------
    tuple
        - H_final : sympy.Matrix
            The final Hamiltonian matrix after solving.
        - S : dict
            A dictionary of solution matrices for each key.

    Examples:
    ---------
    >>> H = Matrix([[1, 0], [0, 2]])
    >>> list_subspaces = [0, 1]
    >>> solver(H, list_subspaces, order=1)
    (Matrix([
    [1, 0],
    [0, 2]]), {})
    """

    from_list_to_key_lenght = lambda l: ('-'.join([str(i) for i in l]), len(l))

    list_keys_full = generate_keys(order)
    keys = rearrange_keys(list_keys_full)
    keys.pop()

    rational_factorial = [Rational(1, factorial(i)) for i in range(order + 1)]

    H_ordered = group_by_order(H)
    elementes_ordered = {str(key): get_matrix(value, list_subspaces) for key, value in H_ordered.items()}

    dim = Mul(*[list_subspaces[i][1] for i in range(len(list_subspaces))])
    zero_matrix = zeros(dim)
    H0_full = elementes_ordered.get('0', zero_matrix)
    H0 = diag(*H0_full.diagonal()) # just diagonal
    H0_p = H0_full - H0 # off-diagonal of zeroth order
    if sum(Abs(H0_p)) != 0 and full_diagonal:
        warn("Complete perturbative diagonalization is impossible as Hamiltonian contains 0th order off-diagonal parameters.") 
        
    Vk_dict = {}
    Bk_dict = {}
    H_final = zeros(dim)
    
    for key, value in H_ordered.items():
        Hk = group_by_diagonal(value)
        if key == 0: # if order is zero include also possible zero order terms outside of diagonal
            H_final += get_matrix(Hk.get(True, 0), list_subspaces) + get_matrix(Hk.get(False, 0), list_subspaces)
            Vk_dict[key] = zero_matrix
            continue
        H_final += get_matrix(Hk.get(True, 0), list_subspaces)
        Vk_dict[key] = get_matrix(Hk.get(False, 0), list_subspaces)

    S = {}

    for key in tqdm(keys):
        k_total, l_total = from_list_to_key_lenght(key)
        order_it = np_sum(key)
        k_last, _ = from_list_to_key_lenght(key[:-1])
        k, _ = from_list_to_key_lenght([key[-1]])

        if l_total == 1:
            continue
            
        if l_total == 2 and key[0] == 0:
            Vk = Vk_dict.get(order_it, zero_matrix)
            Bk = Bk_dict.get(order_it, zero_matrix)

            Vk_plus_Bk = Vk + Bk

            if Vk_plus_Bk == 0:
                S[k] = zero_matrix
                continue
            Sk, symbols_s = get_ansatz(dim)
            expression_to_solve = (H0 * Sk - Sk * H0) + Vk_plus_Bk
            sols = {s : 0 for s in symbols_s}            
            solution = solve(expression_to_solve, symbols_s, dict=True)[0]
            
            sols.update(solution)
            S[k] = Sk.subs(sols)
            
            kth_ord_off_diag = (H0_p * S[k] - S[k] * H0_p) # this is needed in the scenario we have off diagonals of 0 order
            elementes_ordered[k_total] = - Vk_plus_Bk  + kth_ord_off_diag
            
            H_final += kth_ord_off_diag
                
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
