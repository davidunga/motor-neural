from typing import Sequence, Literal, Annotated, Self, Tuple, Any, Set
import numpy as np
from nptyping import NDArray, Shape, Float, Int, UInt
import types


Pair = {
    float: Annotated[Sequence[float], 2]
}

NpVec = {}
NpMat = {}
NpNx2 = {}
alias_types = {int: Int, float: Float}
for type_ in [int, float, Float, Int, UInt, Any]:
    vtype_ = alias_types.get(type_, type_)
    NpVec[type_] = NDArray[Shape["*"], vtype_]
    NpMat[type_] = NDArray[Shape["*, *"], vtype_]
    NpNx2[type_] = NDArray[Shape["*, 2"], vtype_]

Vec = {
    float: list[float] | NpVec[float],
    int: list[int] | NpVec[int]
}


def is_sorted(x: NpVec[Any], order: Literal['a', 'd'] = 'a'):
    if order == 'a':
        return np.all(x[:-1] <= x[1:])
    return np.all(x[:-1] >= x[1:])


def is_type(x: Any, type_):

    if isinstance(type_, type):
        return isinstance(x, type_)

    if isinstance(type_, types.UnionType):
        for arg_type in type_.__args__:
            if is_type(x, arg_type):
                return True
        return False

    if not isinstance(x, type_.__origin__):
        return False

    try:
        arg_types = type_.__args__
    except AttributeError:
        return True

    match type_.__origin__.__name__:
        case "dict":
            return all([is_type(k, arg_types[0]) and is_type(v, arg_types[1]) for k, v in x.items()])
        case "list":
            return all([is_type(xx, arg_types[0]) for xx in x])
        case "tuple":
            return len(arg_types) == len(x) and all([is_type(xx, arg_type) for xx, arg_type in zip(x, arg_types)])
        case other:
            raise TypeError(f"Type {other} is not supported")

