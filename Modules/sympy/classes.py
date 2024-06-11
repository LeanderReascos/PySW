from typing import Union
from sympy.physics.quantum import Operator
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.printing.latex import print_latex
from sympy import Mul, Pow, eye, Matrix, I
from sympy import zeros as sp_zeros
from sympy.physics.quantum.boson import BosonOp
from sympy.core.singleton import S

from sympy.core.numbers import Integer, Float, ImaginaryUnit, One, Half, Rational

from numpy import array, sqrt
from numpy import zeros as np_zeros

from multimethod import multimethod


class RDOperator(Operator):
    @property
    def name(self):
        return self.args[0]

    @property
    def subspace(self):
        return self.args[1]
    
    @property
    def dim(self):
        return self.args[2]
    
    @property
    def matrix(self):
        return Matrix(self.args[3]).reshape(self.dim, self.dim)
    
    def _sympystr(self, printer):
        return printer._print(self.name)
    
    def _latex(self, printer):
        return printer._print(self.name)
    
    def _eval_commutator_RDOperator(self, other, **options):
        if self.subspace != other.subspace:
            return S.Zero
        return self * other - other * self
    
    def _eval_commutator_RDBoson(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDsymbol(self, other, **options):
        return S.Zero
    
    def __new__(cls, name, subspace, dim, matrix):
        return Operator.__new__(cls, name, subspace, dim, matrix)
    
    
        

class RDsymbol(Symbol):

    @property
    def order(self):
        return self._order
    
    def _eval_commutator_RDOperator(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDBoson(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDsymbol(self, other, **options):
        return S.Zero
    
    def __new__(cls, name, *args, order=0):
        obj = Symbol.__new__(cls, name, *args)
        obj._order = order
        return obj

    
    
        
    
class RDBoson(Operator):

    @property
    def name(self):
        return self.args[0]
    @property
    def subspace(self):
        return self.args[1]
    
    @property
    def is_annihilation(self):
        return self.args[2]
    
    def _sympystr(self, printer):
        return printer._print(self.name)
    
    def _latex(self, printer):
        return printer._print(self.name)
    
    def __new__(cls, name=name, subspace=subspace, is_annihilation=True):
        return Operator.__new__(cls, name, subspace, is_annihilation)
    
    def _eval_commutator_RDOperator(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDBoson(self, other, **options):
        if self.subspace != other.subspace:
            return S.Zero
        if self.is_annihilation == other.is_annihilation:
            return S.Zero
        if self.is_annihilation and not other.is_annihilation:
            return S.One
        if not self.is_annihilation and other.is_annihilation:
            return -S.One
        
    def _eval_commutator_RDsymbol(self, other, **options):
        return S.Zero


@multimethod
def get_terms_and_factors(expr: Union[Symbol, Operator, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns tuple of two lists: one containing ordered terms within expr, the second 
        containing lists of factors for each term."""
    return [expr], [[expr]]

@multimethod
def get_terms_and_factors(expr: Pow):
    """Returns tuple of two lists: one containing ordered terms within expr, the second 
        containing lists of factors for each term."""
    pow_base, pow_exp = expr.as_base_exp()
    if isinstance(pow_base, int) and pow_exp > 0:
        return [expr], [[pow_base for _ in range(pow_exp)]]
    return [expr], [[expr]]

@multimethod
def get_terms_and_factors(expr : Expr):
    """Returns tuple of two lists: one containing ordered terms within expr, the second 
        containing lists of factors for each term."""
    expr = expr.expand()
    terms = expr.as_ordered_terms()
    factors_of_terms = []
    for term in terms:
        factors = term.as_ordered_factors()
        factors_list = []
        for f in factors:
            _, f_list = get_terms_and_factors(f)
            factors_list += f_list[0]
        factors_of_terms.append(factors_list)

    return terms, factors_of_terms

class RDBasis():
    def __init__(self, name, subspace, dim):
        self.name = name
        self.subspace = subspace
        self.dim = dim
        matrix_basis = get_gell_mann(dim)
        self._basis = array([RDOperator(name + f'_{{{i}}}', subspace, dim, mat) for i, mat in enumerate(matrix_basis)], dtype=object)
        self.basis_ling_alg_norm  = (self._basis[0].matrix.T.conjugate() * self._basis[0].matrix).trace()

    @multimethod
    def project(self, to_be_projected : RDOperator):
        # Since it is a basis operator, it is already projected. The subspace does not matter.
        return to_be_projected


    @multimethod
    def project(self, to_be_projected : Union[Symbol, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
        return to_be_projected

    @multimethod
    def project(self, to_be_projected : Expr):
        _, factors = get_terms_and_factors(to_be_projected)
        
        result =[] 
        for term in factors:  
            coeff = 1
            ops = 1
            
            for factor in term:
                if isinstance(factor, RDOperator):
                    if factor.subspace != self.subspace:
                        coeff *= factor
                        continue
                    ops *= factor.matrix
                    continue

                coeff *= factor

            result.append(coeff * self.project(ops))
        
        return sum(result)
    
    @multimethod
    def project(self, to_be_projected : Pow):
        base, exp = to_be_projected.as_base_exp()
        if base.has(Operator):
            pass
            


    @multimethod
    def project(self, to_be_projected : Matrix):
        if to_be_projected.shape != (self.dim, self.dim):
            raise ValueError('Matrix to be projected has wrong shape.')

        basis_coeffs = np_zeros(self.dim**2, dtype=object)

        for i, basis in enumerate(self._basis):
            basis_coeffs[i] = (basis.matrix * to_be_projected.T.conjugate()).trace()
        return basis_coeffs.dot(self._basis)  / self.basis_ling_alg_norm
    
def get_gell_mann(dim):
    """Generates the set of generalized Gell-Mann matrices for a given dimension."""
    matrices = [eye(dim)]
    
    # Lambda_1 to Lambda_(n-1)^2
    for i in range(dim):
        for j in range(i + 1, dim):
            # Symmetric Gell-Mann matrices
            symm = sp_zeros(dim, dim)
            symm[i, j] = 1
            symm[j, i] = 1
            matrices.append(symm)
            
            # Anti-symmetric Gell-Mann matrices
            asymm = sp_zeros(dim, dim)
            asymm[i, j] = -1j
            asymm[j, i] = 1j
            matrices.append(asymm)
    
    # Diagonal Gell-Mann matrices
    for k in range(1, dim):
        diag = sp_zeros(dim, dim)
        for i in range(k):
            diag[i, i] = 1
        diag[k, k] = -k
        diag = diag * sqrt(2 / (k * (k + 1)))
        matrices.append(diag)
    
    return matrices