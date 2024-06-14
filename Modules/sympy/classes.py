from typing import Union
from sympy.physics.quantum import Operator, Commutator
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.printing.latex import print_latex
from sympy import Mul, Pow, eye, Matrix, I, sqrt
from sympy import zeros as sp_zeros
from sympy.physics.quantum.boson import BosonOp
from sympy.core.singleton import S

from sympy.core.numbers import Integer, Float, ImaginaryUnit, One, Half, Rational

from numpy import array
from numpy import zeros as np_zeros

from multimethod import multimethod

from itertools import product


class RDOperator(Operator):

    @property
    def name(self):
        return self.args[0]

    @property
    def subspace(self):
        #return self._subspace
        return self.args[1]
    
    @property
    def dim(self):
        #return self._dim
        return self.args[2]
        
    @property
    def matrix(self):
        #return self._matrix
        return Matrix(self.args[3]).reshape(self.dim, self.dim)
    
    
    def add_product_relation(self, other, result):
        self._product_relations[other] = result
    
    def _sympystr(self, printer):
        return printer._print(self.name)
    
    def _latex(self, printer):
        return printer._print(self.name)
    
    def _eval_commutator_RDOperator(self, other, **options):
        if self.subspace != other.subspace:
            return S.Zero
        self_other = self._product_relations.get(other, self * other)
        other_self = other._product_relations.get(self, other * self)
        return (self_other - other_self).expand()
    
    def _eval_commutator_RDBoson(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDsymbol(self, other, **options):
        return S.Zero
    
    def __new__(cls, name, subspace, dim, matrix):
        obj = Operator.__new__(cls, name, subspace, dim, matrix)
        obj._product_relations = {}
        #obj._subspace = subspace
        #obj._dim = dim
        #obj._matrix = matrix
        return obj
    
    
        

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
        return  self.args[2]
        
    @property
    def dim_projection(self):
        return self.args[3]
    
    @property
    def matrix(self):
        return self._matrix
    
    def _sympystr(self, printer):
        return printer._print(self.name)
    
    def _latex(self, printer):
        return printer._print(self.name)
    
    def __new__(cls, name=name, subspace=None, is_annihilation=True, dim_projection=1):
        obj = Operator.__new__(cls, name, subspace, is_annihilation, dim_projection)

        if is_annihilation:
            obj._matrix = sp_zeros(dim_projection, dim_projection)
            for i in range(1, dim_projection):
                obj._matrix[i-1, i] = sqrt(i)
        else:
            obj._matrix = sp_zeros(dim_projection, dim_projection)
            for i in range(1, dim_projection):
                obj._matrix[i, i-1] = sqrt(i)

        return obj
    
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
        self.subspace = Symbol(subspace)
        self.dim = dim
        matrix_basis = get_gell_mann(dim)
        self._basis = array([RDOperator(name + f'_{{{i}}}', subspace, dim, mat) for i, mat in enumerate(matrix_basis)], dtype=object)
        if len(self._basis) == 1:
            self.basis_ling_alg_norm = dim
        else:
            self.basis_ling_alg_norm  = (self._basis[1].matrix.T.conjugate() * self._basis[1].matrix).trace()

        permutations = self._basis[:,None] * self._basis[None,:]
        self.product_relations = {p : self.project(p) for p in permutations.flatten()}

        for rd in self._basis:
            for p in self._basis:
                rd.add_product_relation(p, self.product_relations[rd*p])
            
    def apply_product_relation(self, expr):
        return apply_substitution(expr, self.product_relations)

    @multimethod
    def project(self, to_be_projected : RDOperator):
        # Since it is a basis operator, it is already projected. The subspace does not matter.
        return to_be_projected


    @multimethod
    def project(self, to_be_projected : Union[Symbol, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
        return to_be_projected

    @multimethod
    def project(self, to_be_projected : Expr):
        to_be_projected = to_be_projected.expand()
        _, factors = get_terms_and_factors(to_be_projected)
        result =[] 
        for term in factors:  
            coeff = 1
            ops = 1
            for factor in term:
                c, op = self.project_to_coeff_and_matrix(factor)
                coeff *= c
                ops *= op
            result.append(coeff * self.project(ops))
        return sum(result)
    
    @multimethod
    def project(self, to_be_projected : Pow):
        c, op = self.project_to_coeff_and_matrix(to_be_projected)
        return c * self.project(op)
    
    @multimethod
    def project(self, to_be_projected : Matrix):
        if to_be_projected.shape != (self.dim, self.dim):
            raise ValueError('Matrix to be projected has wrong shape.')

        basis_coeffs = np_zeros(self.dim**2, dtype=object)
        
        for i, basis in enumerate(self._basis):
            basis_coeffs[i] = (to_be_projected * basis.matrix.T.conjugate()).trace()
        
        basis_coeffs /= self.basis_ling_alg_norm
        basis_coeffs[0] *= self.basis_ling_alg_norm / self.dim
        if basis_coeffs[0] == 1:
            return 1
        return basis_coeffs.dot(self._basis)
    @multimethod            
    def project_to_coeff_and_matrix(self, to_be_projected:RDOperator):
        if to_be_projected.subspace != self.subspace:
            return to_be_projected, 1
        return 1, to_be_projected.matrix
    
    @multimethod
    def project_to_coeff_and_matrix(self, to_be_projected:Union[RDBoson, Symbol, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
        return to_be_projected, 1
    
    @multimethod
    def project_to_coeff_and_matrix(self, to_be_projected:Pow):
        to_be_projected = to_be_projected.expand()
        base, exp = to_be_projected.as_base_exp()
        c, op = self.project_to_coeff_and_matrix(base)
        return c**exp, op**exp

    @multimethod
    def project_to_coeff_and_matrix(self, to_be_projected:Expr):
        return to_be_projected, 1
    

    

class RDCompositeBasis:
    def __init__(self, bases : list[RDBasis]):
        self.bases = bases
        self.dim = Mul(*[basis.dim for basis in bases])
        self._basis = array([Mul(*p) for p in product(*[basis._basis for basis in bases])], dtype=object)

    def project(self, to_be_projected):
        result = to_be_projected
        for basis in self.bases:
            result = basis.project(result)
        return result


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
            asymm[i, j] = -I
            asymm[j, i] = I
            matrices.append(asymm)
    
    # Diagonal Gell-Mann matrices
    for k in range(1, dim):
        diag = sp_zeros(dim, dim)
        for i in range(k):
            diag[i, i] = 1
        diag[k, k] = -k
        diag = diag * sqrt(Rational(2 , (k * (k + 1))))
        matrices.append(diag)
    
    return matrices

def apply_substitution(expr, substitution=None):
    if substitution is None:
        return expr
    expr_new = expr.subs(substitution).expand()
    while expr_new != expr:
        expr = expr_new
        expr_new = expr.subs(substitution).expand()
    return expr_new