from sympy.physics.quantum import Operator
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.printing.latex import print_latex
from sympy import eye, Matrix
from sympy.physics.quantum.boson import BosonOp
from sympy.core.singleton import S


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
    
    def __new__(cls, name, subspace, dim, matrix):
        return Operator.__new__(cls, name, subspace, dim, matrix)
    
    def _eval_commutator_RDOperator(self, other, **options):
        if self.subspace != other.subspace:
            return S.Zero
        return self * other - other * self
    
    def _eval_commutator_RDBoson(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDsymbol(self, other, **options):
        return S.Zero


class RDsymbol(Operator):
    @property
    def name(self):
        return self.args[0]

    @property
    def order(self):
        return self.args[1]
    
    
    def _sympystr(self, printer):
        return printer._print(self.name)
    
    def _latex(self, printer):
        return printer._print(self.name)
    
    
    def __new__(cls, name, order = 1):
        return Operator.__new__(cls, sympify(name), sympify(order))
    
    def _eval_commutator_RDOperator(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDBoson(self, other, **options):
        return S.Zero
    
    def _eval_commutator_RDsymbol(self, other, **options):
        return S.Zero
    
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
