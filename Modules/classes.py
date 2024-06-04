from sympy import symbols
from sympy.physics.quantum import Operator
from numpy import count_nonzero, diag, diagonal
from copy import deepcopy
#%%

class Operator:
    def __init__(self, name, subspace = None, is_diagonal = False, mat_representation = None):
        self.representation = Operator(name)
        self.subspace = subspace
        self.is_diagonal = is_diagonal
        self.mat_representation = mat_representation
        
    def __mul__(self, other):
        if type(other) == Term:
            return Term(self) * other
        else:
            Term(self) * Term(other)
            
    def __add__(self, other):
        if type(other) == Term:
            return Term(self) + other
        else:
            Term(self) + Term(other)
            
    def __sub__(self, other):
        if type(other) == Term:
            return Term(self) - other
        else:
            Term(self) - Term(other)
            
    def __neg__(self):
        return -Term(self)

class Coefficient(Operator):
    
    
    
    
    
    
class Term:
    def __init__(self, token = Coefficient(0))
    self.info = dict()
    self.mat_representation = dict()
    self.is_diagonal = dict()
    
    if token.subspace in self.info:
        self.info[operator.subspace].append(token) # if subspace list already exists append
        self.mat_representation[token.subspace] @= token.mat_representation  
    else:
        self.info[token.subspace] = token # else create
        self.mat_representation[token.subspace] = token.mat_representation
        
    self.is_diagonal[operator.subspace] = (count_nonzero(self.mat_representation[operator.subspace] - diag(diagonal(self.mat_representation[operator.subspace]))) == 0)
    self.order = sum([constant.order for constant in self.info["coeff"]])
    
    def __mul__(self, other):
        if type(other) == Term:
            new_term = deepcopy(self)
            
            for key_other in list(other.info.keys):
                if key_other in new_term.info:
                    new_term.info[key_other].append(other.info.[key_other])
                    new_term.mat_representation[key_other] @= other.mat_representation[key_other]
                else:
                    new_term.info[key_other] = other.info.[key_other]
                    new_term.mat_representation[key_other] = other.mat_representation[key_other]
                     
            new_term.is_diagonal = count_nonzero(self.mat_representation - diag(diagonal(self.mat_representation))) == 0
            new_term.order = self.order + other.order
                                              
            return new_term
        return self * Term(other)
    
    def __add__(self, other):
        if type(other) == Expression:
            return Expression(self) + other
        
        return Expression(self) + Expression(other)
        
         

        