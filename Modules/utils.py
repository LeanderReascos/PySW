from sympy import Matrix, symbols
from Modules.classes import Coefficient, Operator

def S(dim, name, order = 1):
    symbols_list = [symbols(name + f"_{i}") for i in range(dim**2)]
    mat_representation =  Matrix([symbols_list[i] for i in range(dim**2)]).reshape(dim, dim)

    S_op  = Operator(name, mat_representation = mat_representation, is_infinite = False, subspace = 'S')
    S_op.subspace = "finite"
    return S_op * Coefficient(f"\\varepsilon_{{({order})}}", order = order), symbols_list

