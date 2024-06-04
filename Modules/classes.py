from typing import Union
from sympy import  symbols, Rational, Mul, eye, Abs
from numpy import array, concatenate
from sympy.physics.quantum import Operator as spOperator
from sympy import diag
from copy import deepcopy

class Operator:
    def __init__(self, name, *, subspace = None, mat_representation = None, is_infinite = False):
        self.representation = spOperator(name) if name != '' else None
        self.subspace = subspace
        self.mat_representation = mat_representation
        self.corresponding_id = eye((mat_representation.shape)[0]) if mat_representation != 1 else None
        self.is_infinite = is_infinite
    
    def __str__(self):
        return self.representation.__str__()
    
    def __repr__(self):
        return self.representation.__repr__()
    
    def _repr_latex_(self):
        return self.representation._repr_latex_()

    def __mul__(self, other):
        if isinstance(other, Term):
            return Term(self) * other
        return Term(self) * Term(other)
            
    def __add__(self, other):
        if type(other) == Term:
            return Term(self) + other
        else:
            return Term(self) + Term(other)
            
    def __sub__(self, other):
        return self + (-other)
            
    def __neg__(self):
        return -Term(self)

class Coefficient(Operator):
    def __init__(self, value : Union[int, float, complex, symbols], *, order = 0):
        super().__init__('', subspace = "coeff", mat_representation=1, is_infinite = False)

        self.representation = symbols(value) if isinstance(value, str) else value
        self.order = order
    
class Term:
    def __init__(self, token = Coefficient(Rational(0))):
        self.info = {"coeff" : [Coefficient(Rational(1))]}
        self.mat_representation = {"coeff" : 1}
        self.diagonal = {"coeff" : True}

        self.is_infinite = token.is_infinite

        self.info[token.subspace] = [token]
        self.mat_representation[token.subspace] = token.mat_representation
    
        self.diagonal[token.subspace] = self.is_diagonal_subspace(token.subspace)
        self.diagonal["total"] = self.is_diagonal()
        self.order = sum([constant.order for constant in self.info["coeff"]])
    
    def __str__(self):
        string_representation_subspace = []
        for _, listt in self.info.items():
            string_representation_subspace.append(Mul(*map(lambda x: x.representation, listt)).__str__())
        
        return " ".join(string_representation_subspace)
    
    def __repr__(self):
        return self.__str__()
    
    def _repr_latex_(self):
        string_representation_subspace = []
        for _, listt in self.info.items():
            string_representation_subspace.append(Mul(*map(lambda x: x.representation, listt))._repr_latex_())
        
        return " ".join(string_representation_subspace)


    def is_diagonal_subspace(self, subspace):
        if subspace == "coeff":
            return True
        return sum(Abs(self.mat_representation[subspace] - diag(*self.mat_representation[subspace].diagonal()))) == 0
    
    def is_diagonal(self):
        return all([self.is_diagonal_subspace(subspace) for subspace in list(self.mat_representation.keys())[1:]])

        
    def __mul__(self, other):

        if isinstance(other, Term):
            new_term = deepcopy(self)
            
            for key, value in other.info.items():
                new_term.info[key] = new_term.info.get(key, []) + value
                new_term.mat_representation[key] = new_term.mat_representation.get(key, other.info[key][0].corresponding_id) @ other.mat_representation[key] if key != "coeff" else new_term.mat_representation.get(key, 1) * other.mat_representation[key]
                new_term.diagonal[key] = new_term.is_diagonal_subspace(key)
                if not new_term.is_infinite:
                    new_term.is_infinite = any(map(lambda v:v.is_infinite, value))

            new_term.diagonal["total"] = new_term.is_diagonal()
            new_term.order = self.order + other.order
                                            
            return new_term
        return self * Term(other)
    
    
    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self) + other
        
        return Expression(self) + Expression(other)
    
    def __neg__(self):
        new_self = deepcopy(self)
        new_self.info["coeff"].append(Coefficient(Rational(-1)))
        return new_self

    def __sub__(self, other):
        return self + (-other)    
    
class Expression:
    def __init__(self, term = Term()):
        self.terms = array([term], dtype = object)
        self.diagonal = term.diagonal
        self.is_infinite = term.is_infinite

    def __str__(self):
        return " + ".join(map(str, self.terms))
    
    def __repr__(self):
        return self.__str__()
    
    def _repr_latex_(self):
        return " + ".join(map(lambda x: x._repr_latex_(), self.terms))

    def __add__(self, other):
        if isinstance(other, Expression):
            new_expression = deepcopy(self)
            new_expression.terms = concatenate((new_expression.terms, other.terms))
            new_expression.diagonal.update({key : self.diagonal.get(key, True) and value for key, value in other.diagonal.items()})
            new_expression.is_infinite = self.is_infinite or other.is_infinite
            return new_expression
        return self + Expression(other)
    
    def __neg__(self):
        new_self = deepcopy(self)
        new_self.terms = [-term for term in new_self.terms]
        return new_self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        if isinstance(other, Expression):
            new_expression = deepcopy(self)
            new_expression.is_infinite = self.is_infinite or other.is_infinite
            return (new_expression.terms[None, :] * other.terms[:, None]).sum()
        return self * Expression(other)
    
    def __or__(self, other):
        if isinstance(other, Expression):
            return self * other - other * self
        return self | Expression(other)
    
    def nested_commutator(self, other, k=1):
        if k == 0:
            return deepcopy(self)
        if k == 1:
            return self | other
        return (self | other).nested_commutator(other, k-1)
    
    




        
        