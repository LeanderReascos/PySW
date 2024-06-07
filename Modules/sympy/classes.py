from sympy.physics.quantum import Operator
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.printing.latex import print_latex
from sympy import eye

class RDOperator(Operator):
    @property
    def name(self):
        return self.args[0]

    @property
    def subspace(self):
        return self.args[1]
    
    @property
    def dim(self):
        """if dim == -1 -> infinite = true"""
        return self.args[2]
    
    @property
    def infinite(self):
        return self.dim == -1
    
    def _sympystr(self, printer):
        return printer._print(self.name)
    
    def _latex(self, printer):
        return printer._print(self.name)
    
    def __new__(cls, name, subspace, dim):
        if subspace == "finite":
            raise ValueError("The subspace 'finite' is reserved for ansatz operators and should not be used for other.")
        return Operator.__new__(cls, name, subspace, dim)


class RDsymbol(Expr):
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
        return Expr.__new__(cls, sympify(name), sympify(order))
