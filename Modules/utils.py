from sympy import Matrix, symbols
from Modules.classes import Coefficient, Operator

def S(dim, name):
    mat_representation =  Matrix([symbols(name + f"_{i}") for i in range(dim**2)]).reshape(dim, dim)
    S_op  = Operator(name, mat_representation = mat_representation, is_infinite = False, subspace = 'S')
    S_op.subspace = "finite"
    return S_op