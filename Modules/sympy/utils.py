from Modules.sympy.classes import *
from sympy.core.numbers import Integer, Float, ImaginaryUnit, One, Half, Rational
from sympy import eye, kronecker_product, Mul, Add

numbers_list = [int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]

def get_terms_and_factors(expr):
    """Returns tuple of two lists: one containing ordered terms within expr, the second 
        containing lists of factors for each term."""
    expr = expr.expand()
    terms = expr.as_ordered_terms()
    factors_of_terms = [term.as_ordered_factors() for term in terms]
    return terms, factors_of_terms

def group_by_order(expr):
    """Returns dict of expressions separated into orders.
    """
    
    if isinstance(expr, RDsymbol):
        return expr.order
    if isinstance(expr, RDOperator) or type(expr) in numbers_list:
        return 0
    terms, factors_of_terms = get_terms_and_factors(expr)
    order_of_terms = [sum([group_by_order(factor) for factor in term ]) for term in factors_of_terms]
    order_separated = dict()
    
    for orderr, term in zip(order_of_terms, terms):
        if orderr not in order_separated:
            order_separated[orderr] = 0
        order_separated[orderr] += term
    return order_separated

def group_by_infinite(expr):
    """Returns dict of expressions separated into subspaces."""
    if isinstance(expr, RDsymbol) or type(expr) in numbers_list:
        return False
    if isinstance(expr, RDOperator):
        return expr.infinite
    terms, factors_of_terms = get_terms_and_factors(expr)
    infinity_of_terms = [bool(sum([group_by_infinite(factor) for factor in term ])) for term in factors_of_terms]
    infinity_separated = dict()
    
    for infinity, term in zip(infinity_of_terms, terms):
        if infinity not in infinity_separated:
            infinity_separated[infinity] = 0
        infinity_separated[infinity] += term
    return infinity_separated

def group_by_diagonal(self):
        diagonals = set([term.diagonal["total"] for term in self.terms])
        return {diagonal : sum([term for term in self.terms if term.diagonal["total"] == diagonal]) for diagonal in diagonals}

def get_finite_identities(expr):
    """Returns dict of sympy matrices for each subspace spanned by expr."""
    if isinstance(expr, RDsymbol) or type(expr) in numbers_list:
        return dict()
    if isinstance(expr, RDOperator):
        if expr.infinite:
            return dict()
        return {expr.subspace : eye(expr.dim)}
    terms, factors_of_terms = get_terms_and_factors(expr)
    finite_subspaces = set([factor.subspace for term in factors_of_terms for factor in term if isinstance(factor, RDOperator)])
    identities = dict()
    for term in factors_of_terms: #"Can we avoid all these for loops?"
        for factor in term:
            if isinstance(factor, RDOperator):
                if factor.subspace in identities:
                    pass
                identities.update(get_finite_identities(factor))
            else:
                get_finite_identities(factor)
    return identities
        
def domain_expansion(expr, subs_dict, subspaces, identities):
    """Returns sympy Matrix of expanded expression.
    expr is expression to expand.
    subs_dict is dictionary containing substitution rules for each operator into matrix form.
    subspaces is ordered list of subspaces (order will indicate in which order to perform kronecker_product).
    identities is list of identity operators in each subspace.
    """
    
    if isinstance(expr, RDsymbol) or type(expr) in numbers_list:
        return expr
    if isinstance(expr, RDOperator):
        if expr.infinite:
            matrices = [identities[subspace] for subspace in subspaces]
            return kronecker_product(*matrices) * expr
        matrices = [identities[subspace] if subspace != expr.subspace else expr.subs(subs_dict) for subspace in subspaces]
        return kronecker_product(*matrices)
    
    terms, factors_of_terms = get_terms_and_factors(expr)
    
    matrices = []
    for term in factors_of_terms: # "Can we avoid all these for loops?"
        matrices.append(Mul(*[domain_expansion(factor, subs_dict, subspaces, identities) for factor in term]))
    return Add(*matrices)

