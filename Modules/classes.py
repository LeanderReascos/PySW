from typing import Union
from sympy import  symbols, Rational, eye, Abs, I, Mul, diag, kronecker_product
from numpy import array, concatenate, prod
from sympy.physics.quantum import Operator as spOperator
from sympy.core.symbol import Symbol
from copy import deepcopy

from multimethod import multimethod

class Operator:
    def __init__(self, name, *, subspace = None, mat_representation = None, is_infinite = False):
        if subspace == "finite":
            raise ValueError("The subspace 'finite' is reserved for ansatz operators and should not be used for other")
        self.name = name
        self.representation = spOperator(name) if name != '' else None
        self.subspace = subspace
        self.mat_representation = mat_representation
        self.corresponding_id = eye((mat_representation.shape)[0]) if not isinstance(mat_representation, int) else 1
        self.is_infinite = is_infinite
        self.dim_of_finite_space = (mat_representation.shape)[0] if not isinstance(mat_representation, int) and not is_infinite else 1

        self.commutation_relation = {name:0}

    def add_commutation_relation(self, other : 'Operator', coefficient):
        self.commutation_relation.update({
            other.name : coefficient
        })
    
    def __str__(self):
        return self.representation.__str__()
    
    def __repr__(self):
        return self.representation.__repr__()
     
    def _repr_latex_(self):
        return self.representation._repr_latex_()
    
    @multimethod
    def __mul__(self, other : 'Operator'):
        return Term(self) * Term(other)

    @multimethod
    def __mul__(self, other : 'Term'):
        return Term(self) * other
    
    @multimethod
    def __mul__(self, other : Union[int, float, complex, Symbol]):
        if other == 0:
            return 0
        return Term(self) * Term(Coefficient(other))
    
    @multimethod
    def __mul__(self, other : 'Expression'):
        return Expression(Term(self)) * other
    
    @multimethod
    def __rmul__(self, other : Union[int, float, complex, Symbol]):
        return self.__mul__(other)
    
    @multimethod
    def __add__(self, other : 'Operator'):
        return Term(self) + Term(other)
    
    @multimethod
    def __add__(self, other : 'Term'):
        return Term(self) + other
    
    @multimethod
    def __add__(self, other : 'Expression'):
        return Expression(Term(self)) + other
    
    @multimethod
    def __add__(self, other : Union[int, float, complex, Symbol]):
        if other == 0:
            return deepcopy(self)
        return Term(self) + Term(Coefficient(other))
    
    __radd__ = __add__
            
    def __sub__(self, other):
        return self + (-other)
            
    def __neg__(self):
        return -Term(self)

    @multimethod
    def __or__(self, other : 'Operator'):
        if self.subspace == other.subspace:
            if self.commutation_relation.get(other.name) is not None:
                return self.commutation_relation[other.name]
            return self * other - other * self
        return 0
    
    @multimethod
    def __or__(self, other : Union[int, float, complex, Symbol]):
        return 0

    @multimethod
    def __ror__(self, other : Union[int, float, complex, Symbol]):
        return 0
    
    @multimethod
    def __or__(self, other : Union['Expression', 'Term']):
        return - (other | self)

    def domain_expansion(self, subspace_identity, subspaces):
        if self.is_infinite:
            raise ValueError("The operator is infinite and cannot be expanded")
        mat_representations = [subspace_identity[subspace] if subspace != self.subspace else self.mat_representation for subspace in subspaces]
        new_operator = Operator(self.name, subspace = "S", mat_representation = kronecker_product(*mat_representations), is_infinite = True )
        new_operator.subspace = "finite"
        return new_operator

class Coefficient(Operator):
    def __init__(self, value : Union[int, float, complex, Symbol], *, order = 0):
        super().__init__('', subspace = "coeff", mat_representation=1, is_infinite = False)
        self.representation = self.create_representation(value)
        self.order = order
    
    @multimethod
    def create_representation(self, value:Symbol):
        return value
    
    @multimethod
    def create_representation(self, value:int):
        return Rational(value)

    @multimethod
    def create_representation(self, value:float):
        return Rational(value).nsimplify()
    
    @multimethod
    def create_representation(self, value:complex):
        return Rational(value.real).nsimplify() + Rational(value.imag).nsimplify() * I
    
    @multimethod
    def create_representation(self, value:str):
        return symbols(value)
    
class Term:
    def __init__(self, token = Coefficient(0)):
        self.info = {"coeff" : [Coefficient(1)]}
        self.mat_representation = {"coeff" : 1}
        self.diagonal = {"coeff" : True}

        self.is_infinite = token.is_infinite

        self.info[token.subspace] = [token]
        self.mat_representation[token.subspace] = token.mat_representation.copy() if not isinstance(token.mat_representation, int) else token.mat_representation
    
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

    def get_infinite(self):
        new_term = Coefficient(1)
        for key in self.info.keys():
            if self.info[key][0].is_infinite:
                new_term *= prod(self.info[key])
                
        return new_term 
    
    def get_finite(self):
        new_term = Coefficient(1)
        for key in self.info.keys():
            if not self.info[key][0].is_infinite:
                new_term *= prod(self.info[key])
                
        return new_term
    
    def get_dim_of_finite_space(self):
        return prod([self.info[key][0].dim_of_finite_space for key in self.info.keys()])
    
    @multimethod
    def __mul__(self, other : 'Operator'):
        return self * Term(other)
    
    @multimethod
    def __mul__(self, other : 'Term'):
        new_term = deepcopy(self)
        
        for key, value in other.info.items():
            new_term.info[key] = new_term.info.get(key, []) + value
            if key != "coeff":
                new_term.mat_representation[key] = new_term.mat_representation.get(key, other.info[key][0].corresponding_id) @ other.mat_representation[key]
            else:
                new_term.mat_representation[key] = new_term.mat_representation.get(key, 1) * other.mat_representation[key]
            new_term.diagonal[key] = new_term.is_diagonal_subspace(key)
            if not new_term.is_infinite:
                new_term.is_infinite = any(map(lambda v:v.is_infinite, value))
        
        new_term.diagonal["total"] = new_term.is_diagonal()
        new_term.order = self.order + other.order
        
        return new_term
    
    @multimethod
    def __mul__(self, other : Union[int, float, complex, Symbol]):
        if other == 0:
            return 0
        return self * Term(Coefficient(other))
    
    @multimethod
    def __mul__(self, other : 'Expression'):
        return Expression(self) * other
    
    @multimethod
    def __rmul__(self, other : Union[int, float, complex, Symbol]):
        return self.__mul__(other)
    
    @multimethod
    def __add__(self, other: 'Operator'):
        return Expression(self) + Expression(Term(other))
    
    @multimethod
    def __add__(self, other: 'Term'):
        return Expression(self) + Expression(other)
    
    @multimethod
    def __add__(self, other: Union[int, float, complex, Symbol]):
        if other == 0:
            return deepcopy(self)
        return Expression(self) + Expression(Term(Coefficient(other)))
    
    @multimethod
    def __add__(self, other: 'Expression'):
        return Expression(self) + other
    
    __radd__ = __add__
    
    def __neg__(self):
        new_self = deepcopy(self)
        new_self.info["coeff"].append(Coefficient(-1))
        return new_self

    def __sub__(self, other):
        return self + (-other)  

    @multimethod
    def __or__(self, other : Union['Operator', 'Term']):
        subspaces = list(self.info.keys())
        if len(subspaces) == 1:
            return 0
        coefficient = prod(self.info['coeff'])
        A = self.info[subspaces[1]]
        if len(subspaces) == 2:
            A_op = A[0]
            B = prod(A[1:])
            return coefficient * (A_op * (B | other) + (A_op | other) * B)  
        A_op = prod(A)
        B = prod([self.info[subspace] for subspace in subspaces[2:]])
        return coefficient * (A_op * (B | other) + (A_op | other) * B)  
        
    @multimethod
    def __or__(self, other : Union[int, float, complex, Symbol]):
        return 0

    @multimethod
    def __ror__(self, other : Union[int, float, complex, Symbol]):
        return 0

    @multimethod
    def __or__(self, other : 'Expression'):
        return Expression(self) | other

    def domain_expansion(self, total_dimesion, subspace_identity, subspaces):
        if self.is_infinite:
            raise ValueError("The term is infinite and cannot be expanded")

        new_ops = []
        for n_order, subspace in enumerate(subspaces):
            ops_list = self.info.get(subspace, [])
            for op in ops_list:
                new_op = op.domain_expansion(subspace_identity, subspaces)
                new_ops.append(new_op)
        return prod(new_ops)


    
class Expression:
    def __init__(self, term = Term()):
        self.terms = array([term], dtype = object)
        self.diagonal = term.diagonal.copy()
        self.is_infinite = term.is_infinite

    def __str__(self):
        return " + ".join(map(str, self.terms))
    
    def __repr__(self):
        return self.__str__()
    
    def _repr_latex_(self):
        return " + ".join(map(lambda x: x._repr_latex_(), self.terms))
    
    @multimethod
    def __mul__(self, other : 'Operator'):
        return self * Expression(Term(other))
    
    @multimethod
    def __mul__(self, other : 'Term'):
        return self * Expression(other)
    
    @multimethod
    def __mul__(self, other : Union[int, float, complex, Symbol]):
        if other == 0:
            return 0
        return self * Expression(Term(Coefficient(other)))
    
    @multimethod
    def __mul__(self, other : 'Expression'):
        new_expression = deepcopy(self)
        new_expression.is_infinite = self.is_infinite or other.is_infinite
        return (new_expression.terms[None, :] * other.terms[:, None]).sum()
    
    @multimethod
    def __rmul__(self, other : Union[int, float, complex, Symbol]):
        return self.__mul__(other)
    
    
    @multimethod
    def __add__(self, other : 'Operator'):
        return self + Expression(Term(other))
    
    @multimethod
    def __add__(self, other : 'Term'):
        return self + Expression(other)
    
    @multimethod
    def __add__(self, other : Union[int, float, complex, Symbol]):
        if other == 0:
            return deepcopy(self)
        return self + Expression(Term(Coefficient(other)))
    
    @multimethod
    def __add__(self, other : 'Expression'):
        new_expression = deepcopy(self)
        new_expression.terms = concatenate((new_expression.terms, other.terms))
        new_expression.diagonal.update({key : self.diagonal.get(key, True) and value for key, value in other.diagonal.items()})
        new_expression.is_infinite = self.is_infinite or other.is_infinite
        return new_expression

    __radd__ = __add__
    
    def __neg__(self):
        new_self = deepcopy(self)
        new_self.terms = [-term for term in new_self.terms]
        return new_self
    
    def __sub__(self, other):
        return self + (-other)
    
    @multimethod
    def __or__(self, other : 'Operator'):
        return self | Expression(Term(other))
    
    @multimethod
    def __or__(self, other : 'Term'):
        return self | Expression(other)
    
    @multimethod
    def __or__(self, other : Union[int, float, complex, Symbol]):
        return 0
    
    @multimethod
    def __ror__(self, other : Union[int, float, complex, Symbol]):
        return 0
    
    @multimethod
    def __or__(self, other : 'Expression'):
        new_expression = deepcopy(self)
        new_expression.is_infinite = self.is_infinite or other.is_infinite
        return (new_expression.terms[None, :] | other.terms[:, None]).sum()

    def nested_commutator(self, other, k=1):
        if k == 0:
            return deepcopy(self)
        if k == 1:
            return self | other
        comm = (self | other)
        if comm != 0:
            return comm.nested_commutator(other, k-1)
        return 0
    
    def group_by_order(self):
        orders = set([term.order for term in self.terms])
        return {order : sum([term for term in self.terms if term.order == order]) for order in orders}
    
    def group_by_diagonal(self):
        diagonals = set([term.diagonal["total"] for term in self.terms])
        return {diagonal : sum([term for term in self.terms if term.diagonal["total"] == diagonal]) for diagonal in diagonals}

    def group_by_infinite(self):
        return {True : sum([term for term in self.terms if term.is_infinite]), False : sum([term for term in self.terms if not term.is_infinite])}
    
    def get_dim_of_finite_space(self):
        return max([term.get_dim_of_finite_space() for term in self.terms])

    def group_by_infinite_terms(self):

        result_dict = {}

        for term in self.terms:
            term_infinite = term.get_infinite()
            term_finite = term.get_finite()

            result_dict[str(term_infinite)] = result_dict.get(str(term_infinite), 0) + term_finite

        return result_dict

    def domain_expansion(self):
        total_finite_dim = self.get_dim_of_finite_space()
        subspaces = []
        subspaces_identity = {}
        finite_terms = []
        infinite_terms = []
        coefficients = []
        for term in self.terms:
            coefficients.append(prod(term.info["coeff"]))
            finite_terms.append(term.get_finite())
            infinite_terms.append(term.get_infinite())
            subspaces += list(finite_terms[-1].info.keys())[1:]
            
        subspaces = set(subspaces)
        for term in finite_terms:
            if len(list(subspaces_identity.keys())) == len(subspaces):
                break
            for subspace, ops in list((term.info.items()))[1:]:
                if subspaces_identity.get(subspace) is not None:
                    continue
                subspaces_identity[subspace] = ops[0].corresponding_id
        
        new_expression = 0

        for coeff, inf_term, finite_term in zip(coefficients, infinite_terms, finite_terms):
            new_finite_term = finite_term.domain_expansion(total_finite_dim, subspaces_identity, subspaces)

            new_expression += coeff * new_finite_term * inf_term

        return new_expression, list(subspaces)
            









        