from sympy.physics.quantum import Operator
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.printing.latex import print_latex
from sympy import eye, kronecker_product



class RDOperator(Operator):
    @property
    def name(self):
        return self.args[0]

    @property
    def subspace(self):
        if subspace == "finite":
            raise ValueError("The subspace 'finite' is reserved for ansatz operators and should not be used for other")
        return self.args[1]

    @property
    def matrix(self):
        return self.args[2]

    @property
    def is_infinite(self):
        return self.args[3]
    
    
    def subspace_identity(self):
        return eye((self.matrix.shape)[0])
    
    def dim(self):
        return (self.matrix.shape)[0] if not is_infinite else 0
    
    def order(self):
        return min([coeff.order for coeff in self.matrix])
    
    def domain_expansion(self, identities, subspaces):
        if self.is_infinite:
            raise ValueError("The operator is infinite and cannot be expanded")
        matrices = [identities[subspace] if subspace != self.subspace else self.matrix for subspace in subspaces]
        new_operator = RDOperator(self.name, subspace = "S", matrix = kronecker_product(*matrices), is_infinite = False )
        new_operator.subspace = "finite"
        return new_operator
    
    
    def __str__(self):
        return self.name.__str__()
    
    def __repr__(self):
        return self.name.__repr__()
     
    def _repr_latex_(self):
        return self.name._repr_latex_()
    
    def __new__(cls, name, subspace, matrix, is_infinite=False):
        return Operator.__new__(cls, name, subspace, matrix, is_infinite)


class RDsymbols(Symbol):
    @property
    def name(self):
        return self.args[0]

    @property
    def order(self):
        return self.args[1]

    def __new__(cls, name, order = 1):
        return Symbol.__new__(cls, name, order)


def sw_is_infinite(expr):
    if not expr.args:
        return False

    for term in expr.args:
        if isinstance(term, SWOperator):
            if term.infinite:
                return True
        else:
            return sw_is_infinite(term)


def sw_is_diagonal_subspace(expr, subspace):
    def diagonal_subspace(expr, subspace):
        if not expr.args:
            return 0

        total = sp.Matrix()
        for term in expr.args:
            if isinstance(term, SWOperator):
                if term.subspace == subspace:
                    total += term.mat
            else:
                total += diagonal_subspace(term, subspace)
        return total

    total = diagonal_subspace(expr, subspace)
    return total == 0


def sw_get_order(expr):
    if not expr.args:
        return 0

    total = 0
    for term in expr.args:
        if isinstance(term, Coefficient):
            total += term.order
        else:
            total += sw_get_order(term)
    return total