from Modules.sympy.classes import RDOperator, RDsymbol, RDBoson, Operator, get_terms_and_factors, apply_substitution
from typing import Union
from sympy.core.numbers import Integer, Float, ImaginaryUnit, One, Half, Rational
from sympy.core.power import Pow
from sympy.core.expr import Expr
from sympy import eye, kronecker_product, Mul, Add, Abs, diag, latex, prod, symbols, Matrix, zeros
from sympy.physics.quantum import Commutator

from numpy import any as np_any
from numpy import sum as np_sum
from numpy import all as np_all
from multimethod import multimethod

numbers_list = [int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]


@multimethod
def group_by_order(expr: Pow):
    """Returns dict of expressions separated into orders."""
    base_order = group_by_order(expr.base)
    if isinstance(base_order, int):
        return  base_order * expr.exp
    return list(base_order.keys())[0] * expr.exp

@multimethod
def group_by_order(expr: RDsymbol):
    """Returns dict of expressions separated into orders.
    """
    return expr.order

@multimethod
def group_by_order(expr: Union[RDBoson, RDOperator, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    return 0

@multimethod
def group_by_order(expr: Expr):
    """Returns dict of expressions separated into orders."""
    terms, factors_of_terms = get_terms_and_factors(expr)
    
    order_separated = {}
    
    for term, factors in zip(terms, factors_of_terms):
        orderr = np_sum(group_by_order(factor) for factor in factors)
        order_separated[orderr] = order_separated.get(orderr, 0) + term
        
    return order_separated

@multimethod
def group_by_infinite(expr: RDBoson):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    return {True : expr}

@multimethod
def group_by_infinite(expr: Union[RDOperator, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    return {False : expr}

@multimethod
def group_by_infinite(expr: Expr):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    terms, factors_of_terms = get_terms_and_factors(expr)

    infinity_separated = dict()

    for term, factors in zip(terms, factors_of_terms):
        infinity = np_any([any(group_by_infinite(factor).keys()) for factor in factors])
        infinity_separated[infinity] = infinity_separated.get(infinity, 0) + term

    return infinity_separated

@multimethod
def group_by_infinite(expr: Pow):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    is_inf = any(group_by_infinite(expr.base).keys())
    return {is_inf : expr}


def group_by_finite(expr):
    """Returns dict of expressions separated into finite and infinite subspaces."""
    result_infinite = group_by_infinite(expr)
    # Swap keys
    return {not key : value for key, value in result_infinite.items()}


@multimethod
def group_by_has_finite(expr: RDOperator):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    return {True : expr}

@multimethod
def group_by_has_finite(expr: Union[RDBoson, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    return {False : expr}

@multimethod
def group_by_has_finite(expr: Expr):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    terms, factors_of_terms = get_terms_and_factors(expr)

    finite_separated = dict()

    for term, factors in zip(terms, factors_of_terms):
        finite = np_any([any(group_by_has_finite(factor).keys()) for factor in factors])
        finite_separated[finite] = finite_separated.get(finite, 0) + term

    return finite_separated

@multimethod
def group_by_has_finite(expr: Pow):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    is_f = any(group_by_has_finite(expr.base).keys())
    return {is_f : expr}


@multimethod
def count_bosons(expr: Union[RDOperator, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns number of bosons in expression."""
    return


@multimethod
def count_bosons(expr: RDBoson):
    """Returns number of bosons in expression."""
    return {expr.subspace : {"annihilation" if expr.is_annihilation else "creation": 1}}

@multimethod
def count_bosons(expr: Pow):
    """Returns number of bosons in expression."""
    expr = expr.expand()
    base, exp = expr.as_base_exp()
    result_count = count_bosons(base)
    if result_count is None:
        return
    result_list = list(result_count.items())
    if len(result_list) == 0:
        return
    subspace, result_count = result_list[0]
    return {subspace : {key: item * exp for key, item in result_count.items()}}

@multimethod
def count_bosons(expr: Expr):
    """Returns number of bosons in expression. Note that this function should receive only *Terms*"""
    expr = expr.expand()
    terms, factors_of_terms = get_terms_and_factors(expr)

    boson_count = dict()
    for term in factors_of_terms:
        for factor in term:
            result_count = count_bosons(factor)
            if result_count is  None:
                continue
            result_list = list(result_count.items())
            if len(result_list) == 0:
                continue
            subspace, result_count = result_list[0]
            for key, item in result_count.items():
                if boson_count.get(subspace) is None:
                    boson_count[subspace] = {key: item}
                    continue
                boson_count.get(subspace).update({key: boson_count.get(subspace, dict()).get(key, 0) + item})

    return boson_count



@multimethod
def group_by_diagonal(expr: RDOperator):
    """Returns dict of expressions separated into diagonal and non-diagonal terms."""
    return {np_sum(Abs(expr.matrix - diag(*expr.matrix.diagonal()))) == 0 : expr}

@multimethod
def group_by_diagonal(expr: Union[RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns dict of expressions separated into diagonal and non-diagonal terms."""
    return {True : expr}

@multimethod
def group_by_diagonal(expr: Pow):
    """Returns dict of expressions separated into diagonal and non-diagonal terms."""
    result = group_by_diagonal(expr.base)
    return {key : value**expr.exp for key, value in result.items()}

@multimethod
def group_by_diagonal(expr: Union[Expr, RDBoson]):
    """Returns dict of expressions separated into diagonal and non-diagonal terms."""

    terms, factors_of_terms = get_terms_and_factors(expr)

    diagonal_separated = dict()

    for term, factors_of_term in zip(terms, factors_of_terms):
        bosons_counts = count_bosons(term)
        is_boson_diagonal = bosons_counts is None or np_all([boson_count.get("creation", 0) == boson_count.get("annihilation", 0) for boson_count in bosons_counts.values()])
        is_finite_diagonal = np_all([list(group_by_diagonal(factor).keys())[0] for factor in factors_of_term if not factor.has(RDBoson)])

        diagonal_separated[is_boson_diagonal and is_finite_diagonal] = diagonal_separated.get(is_boson_diagonal and is_finite_diagonal, 0) + term

    return diagonal_separated


@multimethod
def get_finite_identities(expr: Union[RDBoson, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns dict of sympy matrices for each subspace spanned by expr."""
    return dict()

@multimethod
def get_finite_identities(expr: RDOperator):
    return {expr.subspace : eye(expr.dim)}

@multimethod
def get_finite_identities(expr: Expr):
    """Returns dict of sympy matrices for each subspace spanned by expr."""
    terms, factors_of_terms = get_terms_and_factors(expr)

    identities = dict()
    for term in factors_of_terms:
        for factor in term:
            identities.update(get_finite_identities(factor))
    return identities

def nested_commutator(A, B, k=1):
    if k == 0:
        return A
    if k == 1:
        return Commutator(A, B)

    return Commutator(nested_commutator(A, B, k-1), B)

def group_by_infinite_operators(expr, commutation_relations = None):
    expr = expr.expand()
    infinit_dict = group_by_infinite(expr)
    result_dict = {1: infinit_dict.get(False, 0)}

    infinite_expr = infinit_dict.get(True, 0)
    terms, factors_of_terms = get_terms_and_factors(infinite_expr)

    for term in factors_of_terms:
        result_infinite_term = 1
        result_finite_term = 1
        for factor in term:
            if isinstance(factor, RDBoson):
                result_infinite_term *= factor
                continue
            if isinstance(factor, Pow):
                base, exp = factor.as_base_exp()
                if isinstance(base, RDBoson):
                    result_infinite_term *= base**exp
                    continue

            result_finite_term *= factor

        if commutation_relations is not None:
            result_infinite_term = apply_commutation_relations(result_infinite_term, commutation_relations)
            new_infinite_terms = result_infinite_term.as_ordered_terms()
            for term in new_infinite_terms:
                t, c = term.as_coefficients_dict().popitem()
                result_dict[t] = result_dict.get(t, 0) + c * result_finite_term
            continue

        result_dict[result_infinite_term] = result_dict.get(result_infinite_term, 0) + result_finite_term
    
    return result_dict

def apply_commutation_relations(expr, commutation_relations=None):
    return apply_substitution(expr, commutation_relations)

def expand_commutator(expr):
    expr_expanded = expr.expand(commutator=True)
    while expr_expanded != expr:
        expr = expr_expanded
        expr_expanded = expr.expand(commutator=True)
    return expr_expanded

def group_by_finite_operators(expr):
    expr = expr.expand()
    finite_dict = group_by_has_finite(expr)
    result_dict = {1: finite_dict.get(False, 0)}

    finite_expr = finite_dict.get(True, 0)
    terms, factors_of_terms = get_terms_and_factors(finite_expr)

    for term in factors_of_terms:
        result_infinite_term = 1
        result_finite_term = 1
        for factor in term:
            if isinstance(factor, RDOperator):
                result_finite_term *= factor
                continue
            if isinstance(factor, Pow):
                base, exp = factor.as_base_exp()
                if isinstance(base, RDOperator):
                    result_finite_term *= base**exp
                    continue

            result_infinite_term *= factor
        result_dict[result_finite_term] = result_dict.get(result_finite_term, 0) + result_infinite_term
    
    return result_dict

@ multimethod
def get_matrix(H: RDOperator, list_subspaces):
    return kronecker_product(*[H.matrix if H.subspace == subspace else eye(dim) for subspace, dim in list_subspaces])

@ multimethod
def get_matrix(H: RDBoson, list_subspaces):
    return kronecker_product(*[H.matrix if H.subspace == subspace else eye(dim) for subspace, dim in list_subspaces])

@ multimethod
def get_matrix(H: Union[RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational], list_subspaces):
    return H * kronecker_product(*[eye(dim) for subspace, dim in list_subspaces])

@multimethod
def get_matrix(H: Pow, list_subspaces):
    base, exp = H.as_base_exp()
    return get_matrix(base, list_subspaces) ** exp

@ multimethod
def get_matrix(H: Expr, list_subspaces):
    # list_subspaces : [[subspace, dim], ...]
    H = H.expand()
    terms, factors = get_terms_and_factors(H)
    result_matrix = zeros(prod([dim for subspace, dim in list_subspaces]))
    for term in factors:
        term_matrix = 1
        for factor in term:
            factor_matrix = get_matrix(factor, list_subspaces)
            term_matrix *= factor_matrix
        result_matrix += term_matrix
    return result_matrix

from IPython.display import display, Math

def display_dict(dictionary):
    for key, value in dictionary.items():
        display(Math(f"{latex(key)} : {latex(value)}"))