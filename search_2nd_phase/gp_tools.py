#    -----------------------------------------------------------------
#    This code is copied from open-source software DEAP (https://github.com/DEAP/deap/blob/master/deap/gp.py)
#    and make our customized modifications as needed.
#                                       ——Jiaxu Cui, 2023-12
#    -----------------------------------------------------------------

"""The :mod:`gp` module provides the methods and classes to perform
Genetic Programming with DEAP. It essentially contains the classes to
build a Genetic Program Tree, and the functions to evaluate it.
"""
import ast
import copy
import math
import time
import copyreg
import random
import re
import sys
import types
import warnings
from inspect import isclass

import os
import sys
import contextlib

from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt

import func_timeout
import numpy as np
import numpy.random

from deap import *
import sympy as sp
from func_timeout import func_set_timeout

from torch_scatter import scatter_sum
import torch
import scipy.integrate as spi
import multiprocessing as mp

import os

# from screen_pretrain_knowledge_eq import t_range

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

######################################
# GP Data structure                  #
######################################

# Define the name of type for any types.
__type__ = object


class PrimitiveTree(list):
    """Tree specifically formatted for optimization of genetic programming
    operations. The tree is represented with a list, where the nodes are
    appended, or are assumed to have been appended when initializing an object
    of this class with a list of primitives and terminals e.g. generated with
    the method **gp.generate**, in a depth-first order.
    The nodes appended to the tree are required to have an attribute *arity*,
    which defines the arity of the primitive. An arity of 0 is expected from
    terminals nodes.
    """

    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __setitem__(self, key, val):
        # Check for most common errors
        # Does NOT check for STGP constraints
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError("Invalid slice object (try to assign a %s"
                                 " in a tree of size %d). Even if this is allowed by the"
                                 " list object slice setter, this should not be done in"
                                 " the PrimitiveTree context, as this may lead to an"
                                 " unpredictable behavior for searchSubtree or evaluate."
                                 % (key, len(self)))
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError("Invalid slice assignation : insertion of"
                                 " an incomplete subtree is not allowed in PrimitiveTree."
                                 " A tree is defined as incomplete when some nodes cannot"
                                 " be mapped to any position in the tree, considering the"
                                 " primitives' arity. For instance, the tree [sub, 4, 5,"
                                 " 6] is incomplete if the arity of sub is 2, because it"
                                 " would produce an orphan node (the 6).")
        elif val.arity != self[key].arity:
            raise ValueError("Invalid node replacement with a node of a"
                             " different arity.")
        list.__setitem__(self, key, val)

    def __str__(self):
        """Return the expression in a human readable string.
        """
        string = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)

        return string

    @classmethod
    def from_string(cls, string, pset):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.

        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))

                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)

                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                if not issubclass(type(token), type_):
                    raise TypeError("Terminal {} type {} does not "
                                    "match the expected one: {}."
                                    .format(token, type(token), type_))

                expr.append(Terminal(token, False, type_))

        return cls(expr)

    @classmethod
    def add_mul_to_variables(cls,s):
        
        pattern = re.compile(r'x\d+')
        matches = list(pattern.finditer(s))

        # 由于插入会改变字符串长度，我们需要从后往前处理
        for match in reversed(matches):
            var_start, var_end = match.span()
            var = match.group()

            # 查找前面的 "Mul"
            mul_pos = s.rfind('Mul', 0, var_start)

            needs_wrap = True
            if mul_pos != -1:
                # 检查 "Mul" 和变量之间是否有 ")"
                between = s[mul_pos + 3:var_start]  # "Mul" 长度为 3
                if ')' not in between:
                    needs_wrap = False

            if needs_wrap:
                # 插入 "Mul(1.0," 和 ")"
                s = s[:var_start] + 'Mul(1.0, ' + s[var_start:var_end] + ')' + s[var_end:]

        return s
    ##############################################################
    @classmethod
    def from_string_sympy(cls, string_sympy, pset):  # implemented by Jiaxu Cui at 2023-12-24
        preorder_list = []
        # print(string_sympy)
        # print(string_sympy, sp.simplify(string_sympy))
        # build_tree(sp.simplify(sp.simplify(string_sympy).expand()), preorder_list)
        # print(TreeToDeadStr(string_sympy,converter))
        build_tree(sp.simplify(string_sympy), preorder_list) #简化后的等式 -> sp.simplify(string_sympy)
        # print(preorder_list)
        #用于将 SymPy 表达式转换为自定义的表达式树，并以前序遍历（preorder traversal）的方式存储到 preorder_list 中
        # print(a)
        # print(preorder_list)
        # print("+++++++++++++++++++++++++")
        ind_str = "".join(preorder_list)
        # ind_str = cls.add_mul_to_variables(ind_str)
        # print(cls.add_mul_to_variables(ind_str))

        # print(ind_str)
        # exit(-1)
        return cls.from_string(ind_str, pset)
    @classmethod
    def from_string_sympy_fast(cls, string_sympy, pset):  # implemented by Jiaxu Cui at 2023-12-24
        preorder_list = []
        # safe_string_sympy = safe_sympify(string_sympy)
        # build_tree(safe_string_sympy, preorder_list)
        # build_tree(sp.sympify(string_sympy), preorder_list) #简化后的等式 -> sp.simplify(string_sympy)        
        # build_tree(sp.sympify(string_sympy, evaluate=False), preorder_list)

        build_tree(sp.sympify(string_sympy, evaluate=False), preorder_list)
        ind_str = "".join(preorder_list)
        # print(ind_str)
        # ind_str = cls.add_mul_to_variables(ind_str)
        # print(cls.add_mul_to_variables(ind_str))
        # exit(-1)
        return cls.from_string(ind_str, pset)
    ##############################################################
    @property
    def height(self):
        """Return the height of the tree, or the depth of the
        deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth

    @property
    def root(self):
        """Root of the tree, the element 0 of the list.
        """
        return self[0]

    def searchSubtree(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)


class Primitive(object):
    """Class that encapsulates a primitive and when called with arguments it
    returns the Python code to call the primitive with the arguments.

        >>> pr = Primitive("mul", (int, int), int)
        >>> pr.format(1, 2)
        'mul(1, 2)'
    """
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    def __init__(self, name, args, ret):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret
        args = ", ".join(map("{{{0}}}".format, range(self.arity)))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Terminal(object):
    """Class that encapsulates terminal primitive in expression. Terminals can
    be values or 0-arity functions.
    """
    __slots__ = ('name', 'value', 'ret', 'conv_fct')

    def __init__(self, terminal, symbolic, ret):
        self.ret = ret
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    @property
    def arity(self):
        return 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class MetaEphemeral(type):
    """Meta-Class that creates a terminal which value is set when the
    object is created. To mutate the value, a new object has to be
    generated.
    """
    cache = {}

    def __new__(meta, name, func, ret=__type__, id_=None):
        if id_ in MetaEphemeral.cache:
            return MetaEphemeral.cache[id_]

        if isinstance(func, types.LambdaType) and func.__name__ == '<lambda>':
            warnings.warn("Ephemeral {name} function cannot be "
                          "pickled because its generating function "
                          "is a lambda function. Use functools.partial "
                          "instead.".format(name=name), RuntimeWarning)

        def __init__(self):
            self.value = func()

        attr = {'__init__': __init__,
                'name': name,
                'func': func,
                'ret': ret,
                'conv_fct': repr}

        cls = super(MetaEphemeral, meta).__new__(meta, name, (Terminal,), attr)
        MetaEphemeral.cache[id(cls)] = cls
        return cls

    def __init__(cls, name, func, ret=__type__, id_=None):
        super(MetaEphemeral, cls).__init__(name, (Terminal,), {})

    def __reduce__(cls):
        return (MetaEphemeral, (cls.name, cls.func, cls.ret, id(cls)))


copyreg.pickle(MetaEphemeral, MetaEphemeral.__reduce__)


class PrimitiveSetTyped(object):
    """Class that contains the primitives that can be used to solve a
    Strongly Typed GP problem. The set also defined the researched
    function return type, and input arguments type and number.
    """

    def __init__(self, name, in_types, ret_type, prefix="ARG"):
        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.arguments = []
        # setting "__builtins__" to None avoid the context
        # being polluted by builtins function when evaluating
        # GP expression.
        self.context = {"__builtins__": None}
        self.mapping = dict()
        self.terms_count = 0
        self.prims_count = 0

        self.name = name
        self.ret = ret_type
        self.ins = in_types
        for i, type_ in enumerate(in_types):
            arg_str = "{prefix}{index}".format(prefix=prefix, index=i)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, type_)
            self._add(term)
            self.terms_count += 1

    def renameArguments(self, **kargs):
        """Rename function arguments with new names from *kargs*.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kargs:
                new_name = kargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if issubclass(type_, ret_type):
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if issubclass(prim.ret, type_):
                dict_[type_].append(prim)

    def addPrimitive(self, primitive, in_types, ret_type, name=None):
        """Add a primitive to the set.

        :param primitive: callable object or a function.
        :param in_types: list of primitives arguments' type
        :param ret_type: type returned by the primitive.
        :param name: alternative name for the primitive instead
                     of its __name__ attribute.
        """
        if name is None:
            name = primitive.__name__
        prim = Primitive(name, in_types, ret_type)

        assert name not in self.context or \
               self.context[name] is primitive, \
            "Primitives are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second '%s' primitive." % (name,)

        self._add(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, ret_type, name=None):
        """Add a terminal to the set. Terminals can be named
        using the optional *name* argument. This should be
        used : to define named constant (i.e.: pi); to speed the
        evaluation time when the object is long to build; when
        the object does not have a __repr__ functions that returns
        the code to build the object; when the object class is
        not a Python built-in.

        :param terminal: Object, or a function with no arguments.
        :param ret_type: Type of the terminal.
        :param name: defines the name of the terminal in the expression.
        """
        symbolic = False
        if name is None and callable(terminal):
            name = terminal.__name__

        assert name not in self.context, \
            "Terminals are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            # To support True and False terminals with Python 2.
            self.context[str(terminal)] = terminal

        prim = Terminal(terminal, symbolic, ret_type)
        self._add(prim)
        self.terms_count += 1

    def addEphemeralConstant(self, name, ephemeral, ret_type):
        """Add an ephemeral constant to the set. An ephemeral constant
        is a no argument function that returns a random value. The value
        of the constant is constant for a Tree, but may differ from one
        Tree to another.

        :param name: name used to refers to this ephemeral type.
        :param ephemeral: function with no arguments returning a random value.
        :param ret_type: type of the object returned by *ephemeral*.
        """
        if name not in self.mapping:
            class_ = MetaEphemeral(name, ephemeral, ret_type)
        else:
            class_ = self.mapping[name]
            if class_.func is not ephemeral:
                raise Exception("Ephemerals with different functions should "
                                "be named differently, even between psets.")
            if class_.ret is not ret_type:
                raise Exception("Ephemerals with the same name and function "
                                "should have the same type, even between psets.")

        self._add(class_)
        self.terms_count += 1

    def addADF(self, adfset):
        """Add an Automatically Defined Function (ADF) to the set.

        :param adfset: PrimitiveSetTyped containing the primitives with which
                       the ADF can be built.
        """
        prim = Primitive(adfset.name, adfset.ins, adfset.ret)
        self._add(prim)
        self.prims_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


class PrimitiveSet(PrimitiveSetTyped):
    """Class same as :class:`~deap.gp.PrimitiveSetTyped`, except there is no
    definition of type.
    """

    def __init__(self, name, arity, prefix="ARG"):
        args = [__type__] * arity
        PrimitiveSetTyped.__init__(self, name, args, __type__, prefix)

    def addPrimitive(self, primitive, arity, name=None):
        """Add primitive *primitive* with arity *arity* to the set.
        If a name *name* is provided, it will replace the attribute __name__
        attribute to represent/identify the primitive.
        """
        assert arity > 0, "arity should be >= 1"
        args = [__type__] * arity
        PrimitiveSetTyped.addPrimitive(self, primitive, args, __type__, name)

    def addTerminal(self, terminal, name=None):
        """Add a terminal to the set."""
        PrimitiveSetTyped.addTerminal(self, terminal, __type__, name)

    def addEphemeralConstant(self, name, ephemeral):
        """Add an ephemeral constant to the set."""
        PrimitiveSetTyped.addEphemeralConstant(self, name, ephemeral, __type__)


######################################
# GP Tree compilation functions      #
######################################
def compile(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)


def compileADF(expr, psets):
    """Compile the expression represented by a list of trees. The first
    element of the list is the main tree, and the following elements are
    automatically defined functions (ADF) that can be called by the first
    tree.


    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param psets: List of primitive sets. Each set corresponds to an ADF
                  while the last set is associated with the expression
                  and should contain reference to the preceding ADFs.
    :returns: a function if the main primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    adfdict = {}
    func = None
    for pset, subexpr in reversed(list(zip(psets, expr))):
        pset.context.update(adfdict)
        func = compile(subexpr, pset)
        adfdict.update({pset.name: func})
    return func


######################################
# GP Program generation functions    #
######################################
def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
            (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def genRamped(pset, min_, max_, type_=None):
    """
    .. deprecated:: 1.0
        The function has been renamed. Use :func:`~deap.gp.genHalfAndHalf` instead.
    """
    warnings.warn("gp.genRamped has been renamed. Use genHalfAndHalf instead.",
                  FutureWarning)
    return genHalfAndHalf(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if type(term) is MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


######################################
# GP Crossovers                      #
######################################

def cxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = list(range(1, len(ind1)))
        types2[__type__] = list(range(1, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def cxOnePointLeafBiased(ind1, ind2, termpb):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First typed tree participating in the crossover.
    :param ind2: Second typed tree participating in the crossover.
    :param termpb: The probability of choosing a terminal node (leaf).
    :returns: A tuple of two typed trees.

    When the nodes are strongly typed, the operator makes sure the
    second node type corresponds to the first node type.

    The parameter *termpb* sets the probability to choose between a terminal
    or non-terminal crossover point. For instance, as defined by Koza, non-
    terminal primitives are selected for 90% of the crossover points, and
    terminals for 10%, so *termpb* should be set to 0.1.
    """

    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # Determine whether to keep terminals or primitives for each individual
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if random.random() < termpb else primitive_op
    arity_op2 = terminal_op if random.random() < termpb else primitive_op

    # List all available primitive or terminal types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)

    for idx, node in enumerate(ind2[1:], 1):
        if arity_op2(node.arity):
            types2[node.ret].append(idx)

    common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        # Set does not support indexing
        type_ = random.sample(common_types, 1)[0]
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


######################################
# GP Mutations                       #
######################################
def mutUniform(individual, expr, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,


def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same number of arguments from the :attr:`pset`
    attribute of the individual.

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:  # Terminal
        term = random.choice(pset.terminals[node.ret])
        if type(term) is MetaEphemeral:
            term = term()
        individual[index] = term
    else:  # Primitive
        prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
        individual[index] = random.choice(prims)

    return individual,


def mutEphemeral(individual, mode):
    """This operator works on the constants of the tree *individual*. In
    *mode* ``"one"``, it will change the value of one of the individual
    ephemeral constants by calling its generator function. In *mode*
    ``"all"``, it will change the value of **all** the ephemeral constants.

    :param individual: The normal or typed tree to be mutated.
    :param mode: A string to indicate to change ``"one"`` or ``"all"``
                 ephemeral constants.
    :returns: A tuple of one tree.
    """
    if mode not in ["one", "all"]:
        raise ValueError("Mode must be one of \"one\" or \"all\"")

    ephemerals_idx = [index
                      for index, node in enumerate(individual)
                      if isinstance(type(node), MetaEphemeral)]

    if len(ephemerals_idx) > 0:
        if mode == "one":
            ephemerals_idx = (random.choice(ephemerals_idx),)

        for i in ephemerals_idx:
            individual[i] = type(individual[i])()

    return individual,


def mutInsert(individual, pset):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,


def mutShrink(individual):
    """This operator shrinks the *individual* by choosing randomly a branch and
    replacing it with one of the branch's arguments (also randomly chosen).

    :param individual: The tree to be shrunk.
    :returns: A tuple of one tree.
    """
    # We don't want to "shrink" the root
    if len(individual) < 3 or individual.height <= 1:
        return individual,

    iprims = []
    for i, node in enumerate(individual[1:], 1):
        if isinstance(node, Primitive) and node.ret in node.args:
            iprims.append((i, node))

    if len(iprims) != 0:
        index, prim = random.choice(iprims)
        arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)

        slice_ = individual.searchSubtree(index)
        individual[slice_] = subtree

    return individual,


######################################
#   Our customized modifications     #
######################################


######################################
# GP Tree compilation functions      #
######################################

def compile_torch(expr, pset, constants_symbols):
    """Compile the expression *expr* for torch.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments) + ',' + ','.join(c for c in constants_symbols)
        code = "lambda {args}: {code}".format(args=args, code=code)
        code = code.replace('\"', '')
        code = code.replace('\'', '')
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
# 有限差分
def finite_difference(X, delta_t):
    # X: [t, n ,d]
    diff_X = (X[0:-4] - 8*X[1:-3] + 8*X[3:-1] - X[4:])/(12.0*delta_t)
    return X[2:-2], diff_X
def solve_ivp_with_timeout(eval_func_f, eval_func_g, X0, sparse_A, t_start=0, t_end=1, t_inc=0.01):
    try:
        with stdout_redirected():  # Ignoring warning errors encountered in solving initial value problems
            soluation_Y, t_range = solve_ivp(eval_func_f, eval_func_g, X0, sparse_A, t_start=t_start, t_end=t_end, t_inc=t_inc)
    except func_timeout.exceptions.FunctionTimedOut:
        print('solve_ivp func_timeout ... ')
        t_range = np.arange(t_start, t_end + t_inc, t_inc)
        soluation_Y = np.repeat(X0.reshape(1, -1, X0.shape[-1]), t_range.shape[0],  axis=0)
    return soluation_Y, t_range
def solve_ivp_diff_with_timeout(eval_func_f, eval_func_g, X0, sparse_A, Y ,diff_Y,t_start=0, t_end=1, t_inc=0.01):
    try:
        with stdout_redirected():
            # Ignoring warning errors encountered in solving initial value problems
            print("求差分进入了")
            soluation_Y, t_range = solve_ivp_diff(eval_func_f, eval_func_g,X0, sparse_A, Y, diff_Y,t_start=0, t_end=1, t_inc=0.01)
    except func_timeout.exceptions.FunctionTimedOut:
        print('solve_ivp func_timeout ... ')
        t_range = np.arange(t_start, t_end + t_inc, t_inc)
        soluation_Y = np.repeat(X0.reshape(1, -1, X0.shape[-1]), t_range.shape[0],  axis=0)
    return soluation_Y, t_range


def solve_ivp_diff(eval_func_f, eval_func_g, X0, sparse_A, Y, diff_Y = None, target_idx=None, t_start=0, t_end=1, t_inc=0.01):
    N, x_dim = X0.shape
    if len(sparse_A) == 2:
        row, col = sparse_A
        weights = None
    else:
        row, col, weights = sparse_A
    row = row.astype(int)
    col = col.astype(int)
    #  一维的场景
    def diff_func1_old(x):
        # dx_i(t)/dt = func(x)
        t_len = x.shape[0]
        # N_node = x.shape(1)
        dx_list =[]
        for i in range(t_len):
            x_j = x[i].reshape(-1, x_dim)[row.astype(int)]
            x_i = x[i].reshape(-1, x_dim)[col.astype(int)]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)
            # x_i_j_list.append(x_i_j)
            diff_f = np.array(eval_func_f(*[x[i].reshape(-1, x_dim)[:, m].reshape(-1, 1) for m in range(x_dim)]),
                              dtype=np.float32)
            diff_g = np.array(
                eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, k].reshape(-1, 1) for k in range(x_dim + x_dim)]),
                dtype=np.float32)

            if len(sparse_A) == 3:
                diff_g = diff_g * weights.reshape(-1, 1)

            if len(diff_f.shape) < 2:
                diff_f = x[i].reshape(-1, x_dim)[:, :1]
            if len(diff_g.shape) < 2:
                diff_g = x_i_j[:, :1]
            dX = np.array(diff_f, dtype=np.float32) + \
                 scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)),
                             torch.from_numpy(col).long().view(-1, 1),
                             dim=0,
                             dim_size=x[i].reshape(-1, x_dim).shape[0]).numpy()
            dx_list.append(dX)
        dX = np.array(dx_list)
        # print(dX.shape)
        return dX
    def diff_func1(x):
        t_len = x.shape[0]
        
        # 1. 批量 Reshape：将数据整体转换到 (t_len, N_nodes, x_dim)
        # 使用 -1 自动推断节点数量 N_nodes
        x_reshaped = x.reshape(t_len, -1, x_dim)
        N_nodes = x_reshaped.shape[1]

        # 2. 向量化获取图边特征 (避免在循环中拼接)
        # x_i, x_j 的形状为 (t_len, num_edges, x_dim)
        x_j = x_reshaped[:, row, :]
        x_i = x_reshaped[:, col, :]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)  # (t_len, num_edges, 2 * x_dim)

        # 3. 展平时间维度，一次性传给 eval_func 计算所有时间步
        x_flat = x_reshaped.reshape(-1, x_dim)          # (t_len * N_nodes, x_dim)
        x_i_j_flat = x_i_j.reshape(-1, x_dim * 2)       # (t_len * num_edges, 2 * x_dim)

        # 构造输入并进行函数求值 (无循环计算 f 和 g)
        f_inputs = [x_flat[:, m].reshape(-1, 1) for m in range(x_dim)]
        diff_f = np.array(eval_func_f(*f_inputs), dtype=np.float32)

        g_inputs = [x_i_j_flat[:, k].reshape(-1, 1) for k in range(x_dim * 2)]
        diff_g = np.array(eval_func_g(*g_inputs), dtype=np.float32)

        # 乘上图权重
        if len(sparse_A) == 3:
            # 将 weights 复制 t_len 份，并匹配展平后的形状
            weights_tiled = np.tile(weights, t_len).reshape(-1, 1)
            diff_g = diff_g * weights_tiled

        # 形状防御性处理 (与原逻辑一致)
        if np.isscalar(diff_f) or diff_f.size == 1:
            # 如果是常数，将其填充为与 x_flat 形状一致的常数矩阵
            diff_f = np.full_like(x_flat[:, :1], fill_value=float(diff_f))
        elif len(diff_f.shape) < 2:
            # 如果是 1D 数组但不是常数（虽然在批量处理中较少见），进行 reshape
            diff_f = diff_f.reshape(-1, 1)
        # 对 g 也进行同样的修改
        if np.isscalar(diff_g) or diff_g.size == 1:
            diff_g = np.full_like(x_i_j_flat[:, :1], fill_value=float(diff_g))
        elif len(diff_g.shape) < 2:
            diff_g = diff_g.reshape(-1, 1)

        # 4. 优化版全局 Scatter Sum
        # 为了一次性进行 scatter，我们需要给每个时间步的 col 索引加上跨度偏移量
        # t_offsets 形状: (t_len, 1)
        t_offsets = np.arange(t_len) * N_nodes
        
        # 加上偏移量后展平：使得跨时间步的索引变成了全局唯一的 1D 索引
        col_batched = (col[None, :] + t_offsets[:, None]).flatten()

        # 整个过程只做【一次】 Numpy 到 Torch 的转换
        diff_g_tensor = torch.from_numpy(diff_g)
        col_batched_tensor = torch.from_numpy(col_batched).long().view(-1, 1)

        # 整体执行 scatter_sum
        scattered_g = scatter_sum(
            diff_g_tensor,
            col_batched_tensor,
            dim=0,
            dim_size=t_len * N_nodes
        ).numpy()

        # 5. 组合并还原出期望的形状
        dX_flat = diff_f + scattered_g
        dX = dX_flat.reshape(t_len, N_nodes, -1)  # 还原成类似原先 dx_list 组合后的形状
        
        return dX
    # 外面套一个非线性函数
    def diff_func1_sigmod_new(x):
        t_len = x.shape[0]
        
        # 1. 批量 Reshape：将数据转换到 (t_len, N_nodes, x_dim)
        x_reshaped = x.reshape(t_len, -1, x_dim)
        N_nodes = x_reshaped.shape[1]

        # 2. 计算自动力学 f(x_i)
        x_flat = x_reshaped.reshape(-1, x_dim) # (t_len * N_nodes, x_dim)
        f_inputs = [x_flat[:, m].reshape(-1, 1) for m in range(x_dim)]
        diff_f = np.array(eval_func_f(*f_inputs), dtype=np.float32)

        # 3. 计算聚合后的邻居信息 (即公式中的 sum(A_ij * x_j))
        # 获取源节点特征 x_j: (t_len, num_edges, x_dim)
        x_j = x_reshaped[:, row, :] 
        
        # 应用图权重 A_ij
        if len(sparse_A) == 3:
            # weights 形状为 (num_edges,)，进行广播乘法
            x_j = x_j * weights[None, :, None]

        # 准备进行全局一次性 scatter_sum
        t_offsets = np.arange(t_len) * N_nodes
        col_batched = (col[None, :] + t_offsets[:, None]).flatten()
        
        # 展平加权后的 x_j 准备聚合
        x_j_flat = x_j.reshape(-1, x_dim)
        
        # 将 Numpy 转为 Torch 进行快速聚合计算
        x_j_tensor = torch.from_numpy(x_j_flat)
        col_batched_tensor = torch.from_numpy(col_batched).long().view(-1, 1)

        # 执行聚合：得到每个节点感受到的“总场”，形状 (t_len * N_nodes, x_dim)
        # 这对应论文图 3c 中的 \sum A_ij * x_j [cite: 337, 361]
        aggregated_sum = scatter_sum(
            x_j_tensor,
            col_batched_tensor,
            dim=0,
            dim_size=t_len * N_nodes
        ).numpy()

        # 4. 将聚合后的“总场”传给 eval_func_g 计算整个外部方程
        # 此时 g_inputs 的输入变量是聚合后的节点特征，而非边特征
        
        g_inputs = [aggregated_sum[:, k].reshape(-1, 1) for k in range(x_dim)]
        if len(g_inputs) < 2:
            # 这里的做法取决于你想让符号回归学什么：
            # 如果你想复刻图 3c，g 的输入应该是【聚合后的结果】
            # 如果 eval_func_g 报错缺参数，说明它定义的自变量个数多了
            
            # 临时对齐补齐（仅为跑通，建议重新生成只带 x_dim 个变量的公式）
            while len(g_inputs) < 2: # 假设 lambda 强行要 2 个参数
                g_inputs.append(g_inputs[0])
        diff_g = np.array(eval_func_g(*g_inputs), dtype=np.float32)

        # 5. 形状防御处理与结果组合
        if len(diff_f.shape) < 2:
            diff_f = x_flat[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = aggregated_sum[:, :1]

        # 最终动力学：dX = f(自身) + g(聚合后的邻居) [cite: 337, 407]
        dX_flat = diff_f + diff_g
        dX = dX_flat.reshape(t_len, N_nodes, -1)
        
        return dX
    #  多维的场景
    def diff_func2(x):
        t_len = x.shape[0]
        dx_list = []
        # 获取eval_func_g_list中维度最大的输出的行数
        max_local_size= 0
        max_neighbor_size = 0
        for d in range(x_dim):
            #     # 在这里定义 x_i_j，确保它能够使用
            x_i_full = x[0].reshape(-1, x_dim)  # 假设我们处理第一个时间步的情况，或者可以根据需求调整
            x_j = x_i_full[row]
            x_i = x_i_full[col]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)
            # 局部项维度
            out_local = eval_func_f[d](*[x_i_full[:, m].reshape(-1, 1) for m in range(x_dim)])
            out_local = np.array(out_local, dtype=np.float32).reshape(-1, 1)
            max_local_size = max(max_local_size, out_local.shape[0])

            # 邻接项维度
            out_neighbor = eval_func_g[d](*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
            out_neighbor = np.array(out_neighbor, dtype=np.float32).reshape(-1, 1)
            max_neighbor_size = max(max_neighbor_size, out_neighbor.shape[0])
        for i in range(t_len):
            x_i_full = x[i].reshape(-1, x_dim)
            x_j = x_i_full[row]
            x_i = x_i_full[col]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)

            # === 局部项 ===
            diff_f_dimwise = []
            for d in range(x_dim):
                out_d = eval_func_f[d](*[x_i_full[:, m].reshape(-1, 1) for m in range(x_dim)])
                out_d = np.array(out_d, dtype=np.float32).reshape(-1, 1)
                # 升维到 max_local_size
                if out_d.shape[0] < max_local_size:
                    out_d = np.repeat(out_d, max_local_size // out_d.shape[0], axis=0)
                diff_f_dimwise.append(out_d)
            diff_f = np.concatenate(diff_f_dimwise, axis=1)  # shape: (N_node, x_dim)

            # === 邻接项 ===
            diff_g_dimwise = []
            for d in range(x_dim):
                out_d = eval_func_g[d](*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
                # 确保输出的形状一致
                out_d = np.array(out_d, dtype=np.float32).reshape(-1, 1)
                # 扩展输出的行数使其与最大行数一致，升维到 max_neighbor_size
                if out_d.shape[0] < max_neighbor_size:
                    out_d = np.repeat(out_d, max_neighbor_size // out_d.shape[0], axis=0)
                diff_g_dimwise.append(out_d)
                # print(f"Output shape of eval_func_g_list[{d}] for t={i}: {out_d.shape}")
            diff_g = np.concatenate(diff_g_dimwise, axis=1)  # shape: (E, x_dim)

            if weights is not None:
                diff_g *= weights.reshape(-1, 1)

            # 聚合邻接项
            dX = diff_f + scatter_sum(
                torch.from_numpy(diff_g),
                torch.from_numpy(col).long(),
                dim=0,
                dim_size=x_i_full.shape[0]
            ).numpy()

            dx_list.append(dX)

        return np.array(dx_list)
    # h1n1真实数据
    def diff_func1_targeted(x, target_idx):
        t_len = x.shape[0]
        dx_list = []

        # 只保留与目标国家有关的邻接边
        mask = (col == target_idx)
        col_sel = col[mask]
        row_sel = row[mask]
        weights_sel = weights[mask] if weights is not None else None

        for i in range(t_len):
            x_t = x[i].reshape(-1, x_dim)  # shape: (N, 1)

            # === 局部项 ===
            x_local = x_t[target_idx:target_idx + 1, :]  # shape: (1, 1)
            diff_f = eval_func_f(*[x_local[:, m].reshape(-1, 1) for m in range(x_dim)])
            diff_f = np.array(diff_f, dtype=np.float32).reshape(-1, 1)

            # === 邻接项 ===
            x_j = x_t[row_sel]
            x_i = x_t[col_sel]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)

            # diff_g = eval_func_g(*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
            # diff_g = np.array(diff_g, dtype=np.float32).reshape(-1, 1)

             # 关键修改：确保 diff_g 是 (neighbors,1) 而不是 (1,1)
            diff_g = eval_func_g(*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
            diff_g = np.array(diff_g, dtype=np.float32)  # shape: (neighbors,1)

            if weights_sel is not None:
            # 确保 weights_sel 和 diff_g 形状一致
                weights_reshaped = weights_sel.reshape(-1, 1)  # shape: (neighbors,1)
                diff_g = diff_g * weights_reshaped  # 普通乘法（非原地）


            # if weights_sel is not None:
            #     diff_g *= weights_sel.reshape(-1, 1)

            # 聚合项：因为只取目标国家，所以直接 sum 即可
            g_sum = np.sum(diff_g, axis=0, keepdims=True)  # shape: (1, 1)

            dX = diff_f + g_sum  # shape: (1, 1)
            dx_list.append(dX)

        return np.array(dx_list).reshape(t_len)  # 返回 shape: (T_pred,)
    if x_dim == 1:
    #    start = time.perf_counter()
       pre_diff_Y = diff_func1(Y)
    #    pre_diff_Y = diff_func1_sigmod_new(Y)
    #    end = time.perf_counter()
    #    print(f"运行时间: {end - start:.6f} 秒")
    #    pre_diff_Y = diff_func1_targeted(Y,target_idx)
    else:
        pre_diff_Y= diff_func2(Y)
    return pre_diff_Y
def solve_ivp_diff_noliner(eval_func_f, eval_func_g, eval_func_s, X0, sparse_A, Y, diff_Y=None, target_idx=None, t_start=0,
                   t_end=1,
                   t_inc=0.01):
    """
    修改点：
    1. 增加参数 eval_func_s: 对应公式中的外部非线性函数 S(xi)
    2. 修改 diff_func1: 调整 scatter_sum 与函数求值的顺序
    """
    N, x_dim = X0.shape
    if len(sparse_A) == 2:
        row, col = sparse_A
        weights = None
    else:
        row, col, weights = sparse_A
    row = row.astype(int)
    col = col.astype(int)

    def diff_func1(x):
        t_len = x.shape[0]

        # 1. 批量 Reshape：将数据整体转换到 (t_len, N_nodes, x_dim)
        x_reshaped = x.reshape(t_len, -1, x_dim)
        N_nodes = x_reshaped.shape[1]

        # 2. 向量化获取图边特征
        # x_i, x_j 的形状为 (t_len, num_edges, x_dim)
        x_j = x_reshaped[:, row, :]
        x_i = x_reshaped[:, col, :]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)  # (t_len, num_edges, 2 * x_dim)

        # 3. 展平时间维度，一次性传给 eval_func 计算所有时间步
        x_flat = x_reshaped.reshape(-1, x_dim)  # (t_len * N_nodes, x_dim)
        x_i_j_flat = x_i_j.reshape(-1, x_dim * 2)  # (t_len * num_edges, 2 * x_dim)

        # 计算自动力学 f(x_i)
        f_inputs = [x_flat[:, m].reshape(-1, 1) for m in range(x_dim)]
        diff_f = np.array(eval_func_f(*f_inputs), dtype=np.float32)

        # 计算两两交互项 g(x_i, x_j)
        g_inputs = [x_i_j_flat[:, k].reshape(-1, 1) for k in range(x_dim * 2)]
        diff_g = np.array(eval_func_g(*g_inputs), dtype=np.float32)

        # 乘上图权重 A_ij
        if len(sparse_A) == 3:
            weights_tiled = np.tile(weights, t_len).reshape(-1, 1)
            diff_g = diff_g * weights_tiled

        # 形状防御性处理
        if len(diff_f.shape) < 2:
            diff_f = x_flat[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = x_i_j_flat[:, :1]

        # 4. 执行线性累加 (得到 S 函数括号内的信号 xi)
        # 为了一次性进行 scatter，给每个时间步的 col 索引加上偏移量
        t_offsets = np.arange(t_len) * N_nodes
        col_batched = (col[None, :] + t_offsets[:, None]).flatten()

        diff_g_tensor = torch.from_numpy(diff_g)
        col_batched_tensor = torch.from_numpy(col_batched).long().view(-1, 1)

        # 整体执行 scatter_sum 得到聚合后的中间向量 (t_len * N_nodes, x_dim)
        # aggregated_sum = scatter_sum(
        #     diff_g_tensor,
        #     col_batched_tensor,
        #     dim=0,
        #     dim_size=t_len * N_nodes
        # ).numpy()
        #
        # # 5. 关键修改：将聚合后的结果整体过非线性函数 S(...)
        # # 将聚合后的 aggregated_sum 每一列作为 S 函数的输入
        # s_inputs = [aggregated_sum[:, m].reshape(-1, 1) for m in range(x_dim)]
        # interact_diff = np.array(eval_func_s(*s_inputs), dtype=np.float32)
        # 聚合结果
        aggregated_sum = scatter_sum(
            diff_g_tensor,
            col_batched_tensor,
            dim=0,
            dim_size=t_len * N_nodes
        ).numpy()

        # --- 关键修改：归一化处理 ---
        # 这里的 aggregated_sum 包含了所有时间步，但每个节点在每个时间步的累加都需要除以 N_nodes
        # N_nodes 是你在第一步获取的节点数 (100)
        aggregated_sum = aggregated_sum

        # 5. 将归一化后的结果过非线性函数 S(...)
        s_inputs = [aggregated_sum[:, m].reshape(-1, 1) for m in range(x_dim)]
        interact_diff = np.array(eval_func_s(*s_inputs), dtype=np.float32)

        # 形状防御
        if len(interact_diff.shape) < 2:
            interact_diff = aggregated_sum[:, :1]

        # 6. 组合并还原形状: f(x_i) + S(sum(g))
        dX_flat = diff_f + interact_diff
        dX = dX_flat.reshape(t_len, N_nodes, -1)

        return dX

    # 简化的多维场景适配 (示例仅展示思路)
    def diff_func2(x):
        # 如果需要支持多维，逻辑同上，需遍历维度调用对应的 eval_func_s[d]
        # ... (此处逻辑与 diff_func1 类似，但需处理多个维度函数)
        pass

    if x_dim == 1:
        pre_diff_Y = diff_func1(Y)
    else:
        # 如果 x_dim > 1，建议也按照 diff_func1 的逻辑进行向量化重构
        pre_diff_Y = diff_func1(Y)

    return pre_diff_Y
def solve_ivp_diff_high_order(eval_func_f, eval_func_g, X0, sparse_A, Y, diff_Y=None, target_idx=None, t_start=0, t_end=1,
                   t_inc=0.01):
    N, x_dim = X0.shape

    # ==========================================
    # 1. 调整拓扑解包逻辑 (适应 3-node 索引)
    # ==========================================
    if sparse_A.shape[0] == 3:
        idx1, idx2, idx3 = sparse_A
        weights = None
    elif sparse_A.shape[0] == 4:
        idx1, idx2, idx3, weights = sparse_A
    else:
        # 兼容老版本的二元边逻辑
        idx1, idx2 = sparse_A[0], sparse_A[1]
        idx3 = idx1  # 占位
        weights = sparse_A[2] if sparse_A.shape[0] == 3 else None

    # 确保索引是整数
    idx1 = idx1.astype(int)
    idx2 = idx2.astype(int)
    idx3 = idx3.astype(int)

    def diff_func1(x):
        """批量化的高阶动力学计算 (最推荐的版本)"""
        t_len = x.shape[0]
        x_reshaped = x.reshape(t_len, -1, x_dim)
        N_nodes = x_reshaped.shape[1]

        # ==========================================
        # 2. 修改：获取三个节点的特征并拼接
        # ==========================================
        # 形状均变为 (t_len, num_interactions, x_dim)
        x_target = x_reshaped[:, idx1, :]  # 目标节点 x1
        x_neigh1 = x_reshaped[:, idx2, :]  # 邻居节点 x2
        x_neigh2 = x_reshaped[:, idx3, :]  # 邻居节点 x3

        # 拼接成 (t_len, num_interactions, 3 * x_dim)
        x_triplet = np.concatenate([x_target, x_neigh1, x_neigh2], axis=-1)

        # 3. 展平并计算
        x_flat = x_reshaped.reshape(-1, x_dim)
        x_triplet_flat = x_triplet.reshape(-1, x_dim * 3)  # 注意这里是 3 倍维度

        # 计算局部项 f(x1)
        f_inputs = [x_flat[:, m].reshape(-1, 1) for m in range(x_dim)]
        diff_f = np.array(eval_func_f(*f_inputs), dtype=np.float32)

        # 计算交互项 g(x1, x2, x3)
        # 注意：g_inputs 的长度现在是 x_dim * 3
        g_inputs = [x_triplet_flat[:, k].reshape(-1, 1) for k in range(x_dim * 3)]
        diff_g = np.array(eval_func_g(*g_inputs), dtype=np.float32)

        if weights is not None:
            weights_tiled = np.tile(weights, t_len).reshape(-1, 1)
            diff_g = diff_g * weights_tiled


        if np.isscalar(diff_f) or diff_f.size == 1:
            # 如果是常数，将其填充为与 x_flat 形状一致的常数矩阵
            diff_f = np.full_like(x_flat[:, :1], fill_value=float(diff_f))
        elif len(diff_f.shape) < 2:
            # 如果是 1D 数组但不是常数（虽然在批量处理中较少见），进行 reshape
            diff_f = diff_f.reshape(-1, 1)
        # 对 g 也进行同样的修改
        if np.isscalar(diff_g) or diff_g.size == 1:
            diff_g = np.full_like(x_i_j_flat[:, :1], fill_value=float(diff_g))
        elif len(diff_g.shape) < 2:
            diff_g = diff_g.reshape(-1, 1)

        # ==========================================
        # 4. 优化版全局 Scatter Sum (聚合到 idx1)
        # ==========================================
        t_offsets = np.arange(t_len) * N_nodes
        # 必须聚合到接收信号的目标节点 idx1 上
        idx1_batched = (idx1[None, :] + t_offsets[:, None]).flatten()

        diff_g_tensor = torch.from_numpy(diff_g)
        idx1_batched_tensor = torch.from_numpy(idx1_batched).long().view(-1, 1)

        scattered_g = scatter_sum(
            diff_g_tensor,
            idx1_batched_tensor,
            dim=0,
            dim_size=t_len * N_nodes
        ).numpy()

        # 5. 组合结果
        dX_flat = diff_f + scattered_g
        return dX_flat.reshape(t_len, N_nodes, -1)

    # 如果是多维场景 (x_dim > 1)，以此类推修改 diff_func2
    # 主要是在遍历维度时，将 x_i_j 替换为 x_i_j_k 拼接逻辑

    if x_dim == 1:
        pre_diff_Y = diff_func1(Y)
    else:
        # 这里的逻辑需要根据你 eval_func_f/g 是否是列表来调整
        # 但核心依然是把输入从 2*x_dim 改为 3*x_dim
        pre_diff_Y = diff_func1(Y)  # 暂时假设 diff_func1 已处理多维拼接

    return pre_diff_Y
def solve_ivp_diff_old(eval_func_f, eval_func_g, X0, sparse_A, Y, diff_Y = None, target_idx=None, t_start=0, t_end=1, t_inc=0.01):
    N, x_dim = X0.shape
    if len(sparse_A) == 2:
        row, col = sparse_A
        weights = None
    else:
        row, col, weights = sparse_A
    row = row.astype(int)
    col = col.astype(int)
    #  一维的场景
    def diff_func1_old(x):
        # dx_i(t)/dt = func(x)
        t_len = x.shape[0]
        # N_node = x.shape(1)
        dx_list =[]
        for i in range(t_len):
            x_j = x[i].reshape(-1, x_dim)[row.astype(int)]
            x_i = x[i].reshape(-1, x_dim)[col.astype(int)]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)
            # x_i_j_list.append(x_i_j)
            diff_f = np.array(eval_func_f(*[x[i].reshape(-1, x_dim)[:, m].reshape(-1, 1) for m in range(x_dim)]),
                              dtype=np.float32)
            diff_g = np.array(
                eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, k].reshape(-1, 1) for k in range(x_dim + x_dim)]),
                dtype=np.float32)

            if len(sparse_A) == 3:
                diff_g = diff_g * weights.reshape(-1, 1)

            if len(diff_f.shape) < 2:
                diff_f = x[i].reshape(-1, x_dim)[:, :1]
            if len(diff_g.shape) < 2:
                diff_g = x_i_j[:, :1]
            dX = np.array(diff_f, dtype=np.float32) + \
                 scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)),
                             torch.from_numpy(col).long().view(-1, 1),
                             dim=0,
                             dim_size=x[i].reshape(-1, x_dim).shape[0]).numpy()
            dx_list.append(dX)
        dX = np.array(dx_list)
        # print(dX.shape)
        return dX
    def diff_func1(x):
        t_len = x.shape[0]
        
        # 1. 批量 Reshape：将数据整体转换到 (t_len, N_nodes, x_dim)
        # 使用 -1 自动推断节点数量 N_nodes
        x_reshaped = x.reshape(t_len, -1, x_dim)
        N_nodes = x_reshaped.shape[1]

        # 2. 向量化获取图边特征 (避免在循环中拼接)
        # x_i, x_j 的形状为 (t_len, num_edges, x_dim)
        x_j = x_reshaped[:, row, :]
        x_i = x_reshaped[:, col, :]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)  # (t_len, num_edges, 2 * x_dim)

        # 3. 展平时间维度，一次性传给 eval_func 计算所有时间步
        x_flat = x_reshaped.reshape(-1, x_dim)          # (t_len * N_nodes, x_dim)
        x_i_j_flat = x_i_j.reshape(-1, x_dim * 2)       # (t_len * num_edges, 2 * x_dim)

        # 构造输入并进行函数求值 (无循环计算 f 和 g)
        f_inputs = [x_flat[:, m].reshape(-1, 1) for m in range(x_dim)]
        diff_f = np.array(eval_func_f(*f_inputs), dtype=np.float32)

        g_inputs = [x_i_j_flat[:, k].reshape(-1, 1) for k in range(x_dim * 2)]
        diff_g = np.array(eval_func_g(*g_inputs), dtype=np.float32)

        # 乘上图权重
        if len(sparse_A) == 3:
            # 将 weights 复制 t_len 份，并匹配展平后的形状
            weights_tiled = np.tile(weights, t_len).reshape(-1, 1)
            diff_g = diff_g * weights_tiled

        # 形状防御性处理 (与原逻辑一致)
        if len(diff_f.shape) < 2:
            diff_f = x_flat[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = x_i_j_flat[:, :1]

        # 4. 优化版全局 Scatter Sum
        # 为了一次性进行 scatter，我们需要给每个时间步的 col 索引加上跨度偏移量
        # t_offsets 形状: (t_len, 1)
        t_offsets = np.arange(t_len) * N_nodes
        
        # 加上偏移量后展平：使得跨时间步的索引变成了全局唯一的 1D 索引
        col_batched = (col[None, :] + t_offsets[:, None]).flatten()

        # 整个过程只做【一次】 Numpy 到 Torch 的转换
        diff_g_tensor = torch.from_numpy(diff_g)
        col_batched_tensor = torch.from_numpy(col_batched).long().view(-1, 1)

        # 整体执行 scatter_sum
        scattered_g = scatter_sum(
            diff_g_tensor,
            col_batched_tensor,
            dim=0,
            dim_size=t_len * N_nodes
        ).numpy()

        # 5. 组合并还原出期望的形状
        dX_flat = diff_f + scattered_g
        dX = dX_flat.reshape(t_len, N_nodes, -1)  # 还原成类似原先 dx_list 组合后的形状
        
        return dX
    #  多维的场景
    def diff_func2(x):
        t_len = x.shape[0]
        dx_list = []
        # 获取eval_func_g_list中维度最大的输出的行数
        max_local_size= 0
        max_neighbor_size = 0
        for d in range(x_dim):
            #     # 在这里定义 x_i_j，确保它能够使用
            x_i_full = x[0].reshape(-1, x_dim)  # 假设我们处理第一个时间步的情况，或者可以根据需求调整
            x_j = x_i_full[row]
            x_i = x_i_full[col]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)
            # 局部项维度
            out_local = eval_func_f[d](*[x_i_full[:, m].reshape(-1, 1) for m in range(x_dim)])
            out_local = np.array(out_local, dtype=np.float32).reshape(-1, 1)
            max_local_size = max(max_local_size, out_local.shape[0])

            # 邻接项维度
            out_neighbor = eval_func_g[d](*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
            out_neighbor = np.array(out_neighbor, dtype=np.float32).reshape(-1, 1)
            max_neighbor_size = max(max_neighbor_size, out_neighbor.shape[0])
        for i in range(t_len):
            x_i_full = x[i].reshape(-1, x_dim)
            x_j = x_i_full[row]
            x_i = x_i_full[col]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)

            # === 局部项 ===
            diff_f_dimwise = []
            for d in range(x_dim):
                out_d = eval_func_f[d](*[x_i_full[:, m].reshape(-1, 1) for m in range(x_dim)])
                out_d = np.array(out_d, dtype=np.float32).reshape(-1, 1)
                # 升维到 max_local_size
                if out_d.shape[0] < max_local_size:
                    out_d = np.repeat(out_d, max_local_size // out_d.shape[0], axis=0)
                diff_f_dimwise.append(out_d)
            diff_f = np.concatenate(diff_f_dimwise, axis=1)  # shape: (N_node, x_dim)

            # === 邻接项 ===
            diff_g_dimwise = []
            for d in range(x_dim):
                out_d = eval_func_g[d](*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
                # 确保输出的形状一致
                out_d = np.array(out_d, dtype=np.float32).reshape(-1, 1)
                # 扩展输出的行数使其与最大行数一致，升维到 max_neighbor_size
                if out_d.shape[0] < max_neighbor_size:
                    out_d = np.repeat(out_d, max_neighbor_size // out_d.shape[0], axis=0)
                diff_g_dimwise.append(out_d)
                # print(f"Output shape of eval_func_g_list[{d}] for t={i}: {out_d.shape}")
            diff_g = np.concatenate(diff_g_dimwise, axis=1)  # shape: (E, x_dim)

            if weights is not None:
                diff_g *= weights.reshape(-1, 1)

            # 聚合邻接项
            dX = diff_f + scatter_sum(
                torch.from_numpy(diff_g),
                torch.from_numpy(col).long(),
                dim=0,
                dim_size=x_i_full.shape[0]
            ).numpy()

            dx_list.append(dX)

        return np.array(dx_list)
    # h1n1真实数据
    def diff_func1_targeted(x, target_idx):
        t_len = x.shape[0]
        dx_list = []

        # 只保留与目标国家有关的邻接边
        mask = (col == target_idx)
        col_sel = col[mask]
        row_sel = row[mask]
        weights_sel = weights[mask] if weights is not None else None

        for i in range(t_len):
            x_t = x[i].reshape(-1, x_dim)  # shape: (N, 1)

            # === 局部项 ===
            x_local = x_t[target_idx:target_idx + 1, :]  # shape: (1, 1)
            diff_f = eval_func_f(*[x_local[:, m].reshape(-1, 1) for m in range(x_dim)])
            diff_f = np.array(diff_f, dtype=np.float32).reshape(-1, 1)

            # === 邻接项 ===
            x_j = x_t[row_sel]
            x_i = x_t[col_sel]
            x_i_j = np.concatenate([x_i, x_j], axis=-1)

            # diff_g = eval_func_g(*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
            # diff_g = np.array(diff_g, dtype=np.float32).reshape(-1, 1)

             # 关键修改：确保 diff_g 是 (neighbors,1) 而不是 (1,1)
            diff_g = eval_func_g(*[x_i_j[:, k].reshape(-1, 1) for k in range(x_dim * 2)])
            diff_g = np.array(diff_g, dtype=np.float32)  # shape: (neighbors,1)

            if weights_sel is not None:
            # 确保 weights_sel 和 diff_g 形状一致
                weights_reshaped = weights_sel.reshape(-1, 1)  # shape: (neighbors,1)
                diff_g = diff_g * weights_reshaped  # 普通乘法（非原地）


            # if weights_sel is not None:
            #     diff_g *= weights_sel.reshape(-1, 1)

            # 聚合项：因为只取目标国家，所以直接 sum 即可
            g_sum = np.sum(diff_g, axis=0, keepdims=True)  # shape: (1, 1)

            dX = diff_f + g_sum  # shape: (1, 1)
            dx_list.append(dX)

        return np.array(dx_list).reshape(t_len)  # 返回 shape: (T_pred,)
    if x_dim == 1:
       pre_diff_Y = diff_func1_old(Y)
    #    pre_diff_Y = diff_func1_targeted(Y,target_idx)
    else:
        pre_diff_Y = diff_func2(Y)
    return pre_diff_Y
    # return pre_diff_Y.reshape(diff_Y.shape(0), N, x_dim), diff_Y.reshape(-1, 1)
    # """
    # def diff_func(x, t):
    #     # dx_i(t)/dt = func(x)
    #     x_j = x.reshape(-1, x_dim)[row]
    #     x_i = x.reshape(-1, x_dim)[col]
    #     x_i_j = np.concatenate([x_i, x_j], axis=-1)
    #
    #     # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
    #     diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=1e30)
    #     diff_g = np.nan_to_num(eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=1e30)
    #
    #     if len(diff_f.shape) < 2:
    #         diff_f = x.reshape(-1, x_dim)[:, :1]
    #     if len(diff_g.shape) < 2:
    #         diff_g = x_i_j[:, :1]
    #
    #     # print(diff_f.shape, diff_g.shape)
    #     # assert diff_g.shape[1] == 1
    #     # assert diff_g.shape[0] == x_i_j.shape[0]
    #
    #     # print(np.any(np.isnan(diff_f)), np.any(np.isnan(diff_g)))
    #
    #     dX = np.array(diff_f, dtype=np.float32) + \
    #          scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1,1), dim=0,
    #                      dim_size=x.reshape(-1, x_dim).shape[0]).numpy()
    #
    #     return dX.reshape(-1)
    #
    # t_range = np.arange(t_start, t_end + t_inc, t_inc)
    # # lock.acquire()
    # # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-12, atol=1e-12)
    # # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-9, atol=1e-9)
    # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-3, atol=1e-6)
    #
    # """
    # #按照y的维度调用diff_func
    # # 分配时间步
    # num_steps = int((t_end - t_start) / t_inc*10) + 1
    # t_range = np.linspace(t_start, t_end, num_steps)
    #
    # # 初始化结果
    # X = np.zeros((num_steps, N, x_dim), dtype=np.float32)
    #
    # X[0] = X0
    #
    # def compute_diff(X_t):
    #     x_j = X_t[row]
    #     x_i = X_t[col]
    #     x_i_j = np.concatenate([x_i, x_j], axis=-1)
    #
    #     diff_f = np.array(eval_func_f(*[X_t[:, i].reshape(-1, 1) for i in range(x_dim)]), dtype=np.float32)
    #     diff_g = np.array(eval_func_g(*[x_i_j[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), dtype=np.float32)
    #
    #     if len(sparse_A) == 3:
    #         diff_g = diff_g * weights.reshape(-1, 1)
    #
    #     if len(diff_f.shape) < 2:
    #         diff_f = X_t[:, :1]
    #     if len(diff_g.shape) < 2:
    #         diff_g = x_i_j[:, :1]
    #
    #     dX = np.array(diff_f, dtype=np.float32) + \
    #          scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1, 1),
    #                      dim=0, dim_size=X_t.shape[0]).numpy()
    #
    #     return dX
    # def compute_diff(x):
    #     # dx_i(t)/dt = func(x)
    #     x_j = x.reshape(-1, x_dim)[row]
    #     x_i = x.reshape(-1, x_dim)[col]
    #     x_i_j = np.concatenate([x_i, x_j], axis=-1)
    #
    #     diff_f = np.array(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]),
    #                       dtype=np.float32)
    #     diff_g = np.array(
    #         eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]),
    #         dtype=np.float32)
    #
    #     if len(sparse_A) == 3:
    #         diff_g = diff_g * weights.reshape(-1, 1)
    #
    #     if len(diff_f.shape) < 2:
    #         diff_f = x.reshape(-1, x_dim)[:, :1]
    #     if len(diff_g.shape) < 2:
    #         diff_g = x_i_j[:, :1]
    #
    #     dX = np.array(diff_f, dtype=np.float32) + \
    #          scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1, 1),
    #                      dim=0,
    #                      dim_size=x.reshape(-1, x_dim).shape[0]).numpy()
    #
    #     return dX.reshape(-1)

    # 使用有限差分方法迭代
    # for t in range(1, num_steps):
    #     dX = compute_diff(X[t - 1])
    #     X[t] = X[t - 1] + t_inc * dX
    #
    # X_fd, diff_X = finite_difference(X, t_inc)
    # print("x_fd")
    # print(X_fd)
    #
    #
    # return X_fd, t_range[2:-2].reshape(-1, 1)
def solve_ivp_diff_modified_local(eval_func_f_list, eval_func_g_list, raw_Y, country_select_index, time_select_index, sparse_A):
    T_pred = 44
    N = raw_Y.shape[1]
    dim = 1  # 当前只处理一维

    row, col, weights = sparse_A

    # 初始化全0的预测张量
    pred_output = np.zeros((T_pred, N, dim), dtype=np.float32)

    # 缓存每个国家的邻接子图
    subgraph_cache = {}
    for idx, country_idx in enumerate(country_select_index):
        if country_idx not in subgraph_cache:
            edge_mask = (row == country_idx) | (col == country_idx)
            row_sel = row[edge_mask]
            col_sel = col[edge_mask]
            weights_sel = weights[edge_mask]
            subgraph_cache[country_idx] = (row_sel, col_sel, weights_sel)

    for idx, country_idx in enumerate(country_select_index):
        t0, t1 = time_select_index[idx]
        Y_i = raw_Y[t0:t1, :]  # shape: (44, 130)
        # diff_Y_i = Y_i[1:] - Y_i[:-1]  # shape: (43, 130)
        X0_i = Y_i[0, :].reshape(-1, 1)  # shape: (130, 1)

        # 使用缓存好的子图
        sparse_A_i = subgraph_cache[country_idx]

        # 预测该国家每日新增值（只输出对应列）
        pred_diff = solve_ivp_diff(
            eval_func_f_list,
            eval_func_g_list,
            X0_i,
            sparse_A_i,
            Y_i,
            None,
            target_idx=country_idx,
            t_start=0,
            t_end=T_pred,
            t_inc=1
        )  # shape: (44,)

        # 填入输出张量
        pred_output[:, country_idx, 0] = pred_diff

    return pred_output


eval_func_time_out = 30  ########################################
@func_set_timeout(eval_func_time_out)
def solve_ivp(eval_func_f, eval_func_g, X0, sparse_A, t_start=0, t_end=1, t_inc=0.01):
    N, x_dim = X0.shape
    if len(sparse_A) == 2:
        row, col = sparse_A
    else:
        row, col, weights = sparse_A

    """
    def diff_func(x, t):
        # dx_i(t)/dt = func(x)
        x_j = x.reshape(-1, x_dim)[row]
        x_i = x.reshape(-1, x_dim)[col]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=1e30)
        diff_g = np.nan_to_num(eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=1e30)

        if len(diff_f.shape) < 2:
            diff_f = x.reshape(-1, x_dim)[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = x_i_j[:, :1]

        # print(diff_f.shape, diff_g.shape)
        # assert diff_g.shape[1] == 1
        # assert diff_g.shape[0] == x_i_j.shape[0]

        # print(np.any(np.isnan(diff_f)), np.any(np.isnan(diff_g)))

        dX = np.array(diff_f, dtype=np.float32) + \
             scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1,1), dim=0,
                         dim_size=x.reshape(-1, x_dim).shape[0]).numpy()

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    # lock.acquire()
    # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-12, atol=1e-12)
    # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-9, atol=1e-9)
    New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-3, atol=1e-6)

    """

    def diff_func(x):
        # dx_i(t)/dt = func(x)
        x_j = x.reshape(-1, x_dim)[row]
        x_i = x.reshape(-1, x_dim)[col]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        # diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=0,
        #                       posinf=0, neginf=0)
        # diff_g = np.nan_to_num(
        #    eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=0,
        #    posinf=0, neginf=0)

        diff_f = np.array(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]),
                          dtype=np.float32)
        diff_g = np.array(
            eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]),
            dtype=np.float32)

        if len(sparse_A) == 3:
            diff_g = diff_g * weights.reshape(-1, 1)

        if len(diff_f.shape) < 2:
            diff_f = x.reshape(-1, x_dim)[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = x_i_j[:, :1]

        # print(diff_f.shape, diff_g.shape)
        # assert diff_g.shape[1] == 1
        # assert diff_g.shape[0] == x_i_j.shape[0]

        # print(np.any(np.isnan(diff_f)), np.any(np.isnan(diff_g)))

        dX = np.array(diff_f, dtype=np.float32) + \
             scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1, 1),
                         dim=0,
                         dim_size=x.reshape(-1, x_dim).shape[0]).numpy()

        return dX.reshape(-1)

    s_time = time.time()
    t_range = np.arange(t_start, t_end + t_inc, t_inc)

    def event(t, x):
        if np.isnan(x).any() or np.isinf(x).any():
            return 0.
        return 1.

    event.terminal = True

    sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range,
                        dense_output=True, events=[event], method='RK45', rtol=1e-10,  atol=1e-10)  # , method='RK45', rtol=1e-3, atol=1e-6)
    """
    sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range,
                        dense_output=True, events=[event], method='RK45', rtol=1e-3,
                        atol=1e-6)  # , method='RK45', rtol=1e-3, atol=1e-6)
    """

    """
    sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range,
                        dense_output=True)  # , method='RK45', rtol=1e-3, atol=1e-6)
    """
    # print(sol.status)
    print('time cost of solving ivp: ', time.time() - s_time)
    # print('sol.status is %s' % sol.status)
    if sol.status != 0:
        New_X = np.zeros((len(t_range), N, x_dim)) * np.nan
    else:
        New_X = sol.sol(t_range).T

        New_X = np.nan_to_num(New_X, nan=0, posinf=0, neginf=0)

    """
    if sol.status != 0:
        New_X = np.zeros((len(t_range), N, x_dim))
    else:
        New_X = sol.y.T
    """

    # print(New_X.status)
    # exit(1)
    # New_X = sol.y.T
    # New_X = sol.sol(t_range).T

    # lock.release()

    return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)
def solve_ivp_discrete(eval_func_f, eval_func_g, X0, sparse_A, t_start=0, t_end=1, t_inc=0.01):
    # 真实场景的计算方法 因为积分步长为天数 无法使用有限差分 因此使用累加和的形式 用作离散场景的计算
    N, x_dim = X0.shape
    if len(sparse_A) == 2:
        row, col = sparse_A
    else:
        row, col, weights = sparse_A
    def diff_func(t, x):
        # dx_i(t)/dt = func(x)
        x_j = x.reshape(-1, x_dim)[row]
        x_i = x.reshape(-1, x_dim)[col]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        # diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=0,
        #                       posinf=0, neginf=0)
        # diff_g = np.nan_to_num(
        #    eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=0,
        #    posinf=0, neginf=0)

        diff_f = np.array(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]),
                          dtype=np.float32)
        diff_g = np.array(
            eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]),
            dtype=np.float32)

        if len(sparse_A) == 3:
            diff_g = diff_g * weights.reshape(-1, 1)

        if len(diff_f.shape) < 2:
            diff_f = x.reshape(-1, x_dim)[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = x_i_j[:, :1]

        dX = np.array(diff_f, dtype=np.float32) + \
             scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1, 1),
                         dim=0,
                         dim_size=x.reshape(-1, x_dim).shape[0]).numpy()

        return dX.reshape(-1)

    s_time = time.time()
    # t_range = np.arange(t_start, t_end + t_inc, t_inc)
    t_range = np.arange(t_start, t_end, t_inc)

    def event(t, x):
        if np.isnan(x).any() or np.isinf(x).any():
            return 0.
        return 1.

    event.terminal = True
    # print(f"X0 shape: {X0.shape}")
    # print(f"max(row): {np.max(row)}, max(col): {np.max(col)}")

    def simulate_discrete_dynamics(diff_func, t_range, X0, dt=1.0):
        """
        基于累加方式模拟系统离散动力学
        :param diff_func: callable(t, x_flat) → dx/dt (flat)
        :param t_range: array-like，离散时间步
        :param X0: 初始状态，形状为 (N, x_dim)
        :param dt: 时间步长，默认 1.0
        """
        X0 = X0.reshape(-1)  # 扁平化 (N * x_dim,)
        X = [X0]
        for t in t_range[:-1]:
            x_prev = X[-1]
            dx = diff_func(t, x_prev)  # 仍然返回扁平化结果
            x_next = x_prev + dt * dx
            # x_next = np.maximum(x_next, 0.0)
            # x_next = np.maximum(x_next, x_prev)
            X.append(x_next)
        return np.stack(X)  # 返回形状 (T, N * x_dim)
    # print(diff_func)
    # 原来是算积分 现在因为真实数据集是连续的无法算积分改成 累加和
    New_X = simulate_discrete_dynamics(diff_func, t_range, X0, dt=t_inc)
    # sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range,
    #                     dense_output=True, events=[event], method='RK45', rtol=1e-10,  atol=1e-10)  # , method='RK45', rtol=1e-3, atol=1e-6)

    # sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range,
    #                     dense_output=True, events=[event], method='LSODA', rtol=1e-3,
    #                     atol=1e-6)  # , method='RK45', rtol=1e-3, atol=1e-6)
    # print(New_X.reshape(len(t_range), N, x_dim).shape)
    return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)

# def solve_ivp(eval_func_f, eval_func_g, X0, sparse_A, t_start=0, t_end=1, t_inc=0.01):
#     N, x_dim = X0.shape
#     row, col = sparse_A
#
#     """
#     def diff_func(x, t):
#         # dx_i(t)/dt = func(x)
#         x_j = x.reshape(-1, x_dim)[row]
#         x_i = x.reshape(-1, x_dim)[col]
#         x_i_j = np.concatenate([x_i, x_j], axis=-1)
#
#         # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
#         diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=1e30)
#         diff_g = np.nan_to_num(eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=1e30)
#
#         if len(diff_f.shape) < 2:
#             diff_f = x.reshape(-1, x_dim)[:, :1]
#         if len(diff_g.shape) < 2:
#             diff_g = x_i_j[:, :1]
#
#         # print(diff_f.shape, diff_g.shape)
#         # assert diff_g.shape[1] == 1
#         # assert diff_g.shape[0] == x_i_j.shape[0]
#
#         # print(np.any(np.isnan(diff_f)), np.any(np.isnan(diff_g)))
#
#         dX = np.array(diff_f, dtype=np.float32) + \
#              scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1,1), dim=0,
#                          dim_size=x.reshape(-1, x_dim).shape[0]).numpy()
#
#         return dX.reshape(-1)
#
#     t_range = np.arange(t_start, t_end + t_inc, t_inc)
#     # lock.acquire()
#     # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-12, atol=1e-12)
#     # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-9, atol=1e-9)
#     New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-3, atol=1e-6)
#
#     """
#
#     def diff_func(t, x):
#         # dx_i(t)/dt = func(x)
#         x_j = x.reshape(-1, x_dim)[row]
#         x_i = x.reshape(-1, x_dim)[col]
#         x_i_j = np.concatenate([x_i, x_j], axis=-1)
#
#         # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
#         diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=0,
#                                posinf=0, neginf=0)
#         diff_g = np.nan_to_num(
#             eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=0,
#             posinf=0, neginf=0)
#
#         if len(diff_f.shape) < 2:
#             diff_f = x.reshape(-1, x_dim)[:, :1]
#         if len(diff_g.shape) < 2:
#             diff_g = x_i_j[:, :1]
#
#         # print(diff_f.shape, diff_g.shape)
#         # assert diff_g.shape[1] == 1
#         # assert diff_g.shape[0] == x_i_j.shape[0]
#
#         # print(np.any(np.isnan(diff_f)), np.any(np.isnan(diff_g)))
#
#         dX = np.array(diff_f, dtype=np.float32) + \
#              scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float32)), torch.from_numpy(col).long().view(-1, 1),
#                          dim=0,
#                          dim_size=x.reshape(-1, x_dim).shape[0]).numpy()
#
#         return dX.reshape(-1)
#
#     s_time = time.time()
#     t_range = np.arange(t_start, t_end + t_inc, t_inc)
#
#     def event(t, x):
#         if np.isnan(x).any() or np.isinf(x).any():
#             return 0.
#         return 1.
#
#     event.terminal = True
# #events=[event],
#     sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range,
#                         dense_output=True, rtol=1e-9, atol=1e-9)  # , method='RK45')
#
#     # sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range,
#     #                     dense_output=True)  # , method='RK45', rtol=1e-3, atol=1e-6)
#     print('time cost of solving ivp: ',time.time() - s_time)
#     # print('sol.status is %s' % sol.status)
#     if sol.status != 0:
#         New_X = np.zeros((len(t_range), N, x_dim))
#     else:
#         New_X = sol.sol(t_range).T
#
#         New_X = np.nan_to_num(New_X, nan=0, posinf=0, neginf=0)
#
#     """
#     if sol.status != 0:
#         New_X = np.zeros((len(t_range), N, x_dim))
#     else:
#         New_X = sol.y.T
#     """
#
#     # print(New_X.status)
#     # exit(1)
#     # New_X = sol.y.T
#     # New_X = sol.sol(t_range).T
#
#     # lock.release()
#
#     return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def print_expression(e, level=0):
    spaces = " " * level
    if isinstance(e, (sp.Symbol, sp.Number)):
        print(spaces + str(e))
        return
    if len(e.args) > 0:
        print(spaces + e.func.__name__)
        for arg in e.args:
            print_expression(arg, level + 1)
    else:
        print(spaces + e.func.__name__)

def build_tree(expr, preorder_list,parent = None):#一个递归函数 转换成自定义的表达式树 并以前序遍历的方式存储
    if expr.func.__name__ == 'Add':#a + b + c ——> ['Add(', 'Add(', 'a', ', ', 'b', ')', ', ', 'c', ')']
        # or expr.func.__name__ == 'Mul'
        for _ in range(len(expr.args) - 1):
            preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list, parent=expr)
        for i in range(len(expr.args) - 1):
            preorder_list.append(', ')
            build_tree(expr.args[i + 1], preorder_list, parent=expr)
            preorder_list.append(')')
    elif expr.func.__name__ == 'Sub': #a - b ——> ['Sub(', 'a', ', ', 'b', ')']
        preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list, parent=expr)
        preorder_list.append(', ')
        build_tree(expr.args[1], preorder_list,parent=expr)
        preorder_list.append(')')
    elif expr.func.__name__ == 'Div':
        preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list,parent=expr)
        preorder_list.append(', ')
        build_tree(expr.args[1], preorder_list,parent=expr)
        preorder_list.append(')')
    elif expr.func.__name__ == 'Mul':
        # 分离分子和分母（处理形如 a * b^{-1} 的情况）
        numerators = []
        denominators = []
        for arg in expr.args:
            if arg.func.__name__ == 'Pow' and str(arg.args[1]) == '-1':
                denominators.append(arg.args[0])  # 分母部分
            else:
                numerators.append(arg)  # 分子部分

        # 如果有分母部分，转换为 Div 结构
        if denominators:
            preorder_list.append('Div(')

            # 处理分子部分的乘积
            if len(numerators) > 0:
                if len(numerators) > 1:
                    for _ in range(len(numerators) - 1):
                        preorder_list.append('Mul(')
                    build_tree(numerators[0], preorder_list, parent=expr)
                    for i in range(len(numerators) - 1):
                        preorder_list.append(', ')
                        build_tree(numerators[i + 1], preorder_list, parent=expr)
                        preorder_list.append(')')
                else:
                    build_tree(numerators[0], preorder_list, parent=expr)
            else:
                preorder_list.append('1')  # 分子为1如果没有其他因子

            preorder_list.append(', ')

            # 处理分母部分的乘积
            if len(denominators) > 1:
                for _ in range(len(denominators) - 1):
                    preorder_list.append('Mul(')
                build_tree(denominators[0], preorder_list, parent=expr)
                for i in range(len(denominators) - 1):
                    preorder_list.append(', ')
                    build_tree(denominators[i + 1], preorder_list, parent=expr)
                    preorder_list.append(')')
            else:
                build_tree(denominators[0], preorder_list, parent=expr)

            preorder_list.append(')')

        # 如果没有分母部分，正常处理为嵌套 Mul
        else:
            for _ in range(len(expr.args) - 1):
                preorder_list.append('Mul(')
            build_tree(expr.args[0], preorder_list, parent=expr)
            for i in range(len(expr.args) - 1):
                preorder_list.append(', ')
                build_tree(expr.args[i + 1], preorder_list, parent=expr)
                preorder_list.append(')')
    # elif expr.func.__name__ == 'Pow': #x**3 : ——> ['Mul(', 'Mul(', 'x', ', ', 'x', ')', ', ', 'x', ')']
    #     preorder_list.append('%s(' % expr.func.__name__)
    #     build_tree(expr.args[0], preorder_list,parent=expr)
    #     preorder_list.append(', ')
    #     build_tree(expr.args[1], preorder_list,parent=expr)
    #     preorder_list.append(')')
    # elif expr.func.__name__ == 'Pow': #x**3 : ——> ['Mul(', 'Mul(', 'x', ', ', 'x', ')', ', ', 'x', ')']
    #     # else:  # 如果指数部分是常数或者简单的数值
    #     if re.match(r"^-?\d+(\.\d+)?$", str(expr.args[1])):
    #         if float(expr.args[1]) % 1 == 0:
    #             if expr.args[1] < 0:
    #                 preorder_list.append('Div(')
    #                 preorder_list.append('1')
    #                 preorder_list.append(', ')
    #                 for _ in range(int(abs(expr.args[1])) - 1):
    #                     preorder_list.append('Mul(')
    #                 build_tree(expr.args[0], preorder_list,parent=expr)
    #                 for i in range(int(abs(expr.args[1])) - 1):
    #                     preorder_list.append(', ')
    #                     build_tree(expr.args[0], preorder_list,parent=expr)
    #                     preorder_list.append(')')
    #                 preorder_list.append(')')
    #             elif expr.args[1] == 0:
    #                 preorder_list.append('1')
    #             elif expr.args[1] > 0:
    #                 for _ in range(int(expr.args[1]) - 1):
    #                     preorder_list.append('Mul(')
    #                 build_tree(expr.args[0], preorder_list, parent=expr)
    #                 for i in range(int(expr.args[1]) - 1):
    #                     preorder_list.append(', ')
    #                     build_tree(expr.args[0], preorder_list, parent=expr)
    #                     preorder_list.append(')')
    #             else:
    #                 print('Unknown %s in Pow' % expr.args[1])
    #                 exit(1)
    #         else:
    #             preorder_list.append('Pow(')
    #             build_tree(expr.args[0], preorder_list, parent=expr)
    #             preorder_list.append(', ')
    #             build_tree(expr.args[1], preorder_list, parent=expr)  # 对指数部分也递归处理
    #             preorder_list.append(')')
    #     else:
    #         # preorder_list.append('Pow(')
    #         preorder_list.append('Pow(%s)' % expr.args[0])
    #         # build_tree(expr.args[0], preorder_list)  # 先处理底数
    #         preorder_list.append(', ')
    #         build_tree(expr.args[1], preorder_list, parent=expr)  # 再处理指数部分（可能是包含变量的表达式）
    #         preorder_list.append(')')
    elif expr.func.__name__ == 'Pow':
        if re.match(r"^-?\d+(\.\d+)?$", str(expr.args[1])):
            if float(expr.args[1]) % 1 == 0:
                if expr.args[1] < 0:
                    # For negative exponents, directly create a Div structure
                    preorder_list.append('Div(')
                    preorder_list.append('1')
                    preorder_list.append(', ')
                    # Handle the base with positive exponent
                    for _ in range(int(abs(expr.args[1])) - 1):
                        preorder_list.append('Mul(')
                    build_tree(expr.args[0], preorder_list, parent=expr)
                    for i in range(int(abs(expr.args[1])) - 1):
                        preorder_list.append(', ')
                        build_tree(expr.args[0], preorder_list, parent=expr)
                        preorder_list.append(')')
                    preorder_list.append(')')
                elif expr.args[1] == 0:
                    preorder_list.append('1')
                elif expr.args[1] > 0:
                    for _ in range(int(expr.args[1]) - 1):
                        preorder_list.append('Mul(')
                    build_tree(expr.args[0], preorder_list, parent=expr)
                    for i in range(int(expr.args[1]) - 1):
                        preorder_list.append(', ')
                        build_tree(expr.args[0], preorder_list, parent=expr)
                        preorder_list.append(')')
            else:
                # Non-integer exponent
                preorder_list.append('Pow(')
                build_tree(expr.args[0], preorder_list, parent=expr)
                preorder_list.append(', ')
                build_tree(expr.args[1], preorder_list, parent=expr)
                preorder_list.append(')')
        else:
            # Symbolic exponent
            preorder_list.append('Pow(')
            build_tree(expr.args[0], preorder_list, parent=expr)
            preorder_list.append(', ')
            build_tree(expr.args[1], preorder_list, parent=expr)
            preorder_list.append(')')

    elif expr.func.__name__ == 'exp' or expr.func.__name__ == 'sin' or expr.func.__name__ == 'cos' or expr.func.__name__ == 'tan' or expr.func.__name__ == 'tanh' or expr.func.__name__ == 'sinh' or expr.func.__name__ == 'cosh' or expr.func.__name__ == 'log' or expr.func.__name__ == 'Abs':
        preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list, parent=expr)
        preorder_list.append(')')
    # elif isinstance(expr, (sp.Symbol, sp.Number)):
    #     # 判断是否已有乘法表达式，如果没有则添加
    #     if isinstance(expr, sp.Symbol):
    #         # 这里判断是否已经有常数乘法前缀
    #         if len(preorder_list) > 0 and preorder_list[-1].startswith("Mul("):
    #             preorder_list.append(str(expr))  # 如果已经有 Mul，则直接添加变量
    #         else:
    #             preorder_list.append("Mul(1.0,")  # 如果没有，则添加 Mul(1.0,
    #             preorder_list.append(str(expr))
    #             preorder_list.append(")")
    #     else:
    #         preorder_list.append(str(expr))  # 直接添加常量或数值
    # 2025 4.9 修改 为了将构建的树中每个变量前都乘上一个系数 以便优化常数
    elif isinstance(expr, (sp.Symbol, sp.Number)):
        # if isinstance(expr, sp.Symbol):  # 如果是符号变量
        #     preorder_list.append('1.0 * %s' % expr)  # 为符号变量添加乘常量
        # else:
        preorder_list.append(str(expr))  # 直接添加常量或数值
    # elif isinstance(expr, (sp.Symbol, sp.Number)):
    #     if isinstance(expr, sp.Symbol):
    #         # 如果父节点是 Mul 且 Mul 内的一个参数是数字，则该符号已经是 Mul 的一部分，不加 Mul(1.0, ...)
    #         if isinstance(parent, sp.Mul):
    #             # 如果当前符号是 Mul 的直接参数，且 Mul 内已有数字（常数），则不加 Mul(1.0, ...)
    #             if any(isinstance(arg, (sp.Number, sp.Float, sp.Integer)) for arg in parent.args):
    #                 preorder_list.append(str(expr))
    #             else:
    #                 preorder_list.append('Mul(1.0,')
    #                 preorder_list.append(str(expr))
    #                 preorder_list.append(')')
    #         else:
    #             preorder_list.append('Mul(1.0,')
    #             preorder_list.append(str(expr))
    #             preorder_list.append(')')
    #     else:
    #         preorder_list.append(str(expr))
    elif expr is sp.zoo:
        preorder_list.append(str(1.0))
    elif expr.func.__name__ == 'Pi':
        preorder_list.append(str(3.1415926))
    elif expr.func.__name__ == 'I':
        preorder_list.append(str(0))
    elif expr.func.__name__ == 're':
        preorder_list.append(str(1))
    elif expr.func.__name__ == 'e':
        preorder_list.append(str(2.718281828459))
    elif expr.func.__name__ == 'floor':
        preorder_list.append(str(1))
    elif expr.func.__name__ == 'Exp1' or expr.func.__name__ == 'E':
        preorder_list.append(str(2.718281828))
    else:
        print('Unknown %s' % expr.func.__name__)
        exit(1)
# def build_tree(expr, preorder_list):#一个递归函数 转换成自定义的表达式树 并以前序遍历的方式存储
#     if expr.func.__name__ == 'Add':#a + b + c ——> ['Add(', 'Add(', 'a', ', ', 'b', ')', ', ', 'c', ')']
#         for _ in range(len(expr.args) - 1):
#             preorder_list.append('%s(' % expr.func.__name__)
#         build_tree(expr.args[0], preorder_list)
#         for i in range(len(expr.args) - 1):
#             preorder_list.append(', ')
#             build_tree(expr.args[i + 1], preorder_list)
#             preorder_list.append(')')
#     elif expr.func.__name__ == 'Sub': #a - b ——> ['Sub(', 'a', ', ', 'b', ')']
#         preorder_list.append('%s(' % expr.func.__name__)
#         build_tree(expr.args[0], preorder_list)
#         preorder_list.append(', ')
#         build_tree(expr.args[1], preorder_list)
#         preorder_list.append(')')
#     elif expr.func.__name__ == 'Div':
#         preorder_list.append('%s(' % expr.func.__name__)
#         build_tree(expr.args[0], preorder_list)
#         preorder_list.append(', ')
#         build_tree(expr.args[1], preorder_list)
#         preorder_list.append(')')
#     elif expr.func.__name__ == 'Mul':
#         # 分离分子和分母（处理形如 a * b^{-1} 的情况）
#         numerators = []
#         denominators = []
#         for arg in expr.args:
#             if arg.func.__name__ == 'Pow' and str(arg.args[1]) == '-1':
#                 denominators.append(arg.args[0])  # 分母部分
#             else:
#                 numerators.append(arg)  # 分子部分

#         # 如果有分母部分，转换为 Div 结构
#         if denominators:
#             preorder_list.append('Div(')

#             # 处理分子部分的乘积
#             if len(numerators) > 0:
#                 if len(numerators) > 1:
#                     for _ in range(len(numerators) - 1):
#                         preorder_list.append('Mul(')
#                     build_tree(numerators[0], preorder_list)
#                     for i in range(len(numerators) - 1):
#                         preorder_list.append(', ')
#                         build_tree(numerators[i + 1], preorder_list)
#                         preorder_list.append(')')
#                 else:
#                     build_tree(numerators[0], preorder_list)
#             else:
#                 preorder_list.append('1')  # 分子为1如果没有其他因子

#             preorder_list.append(', ')

#             # 处理分母部分的乘积
#             if len(denominators) > 1:
#                 for _ in range(len(denominators) - 1):
#                     preorder_list.append('Mul(')
#                 build_tree(denominators[0], preorder_list)
#                 for i in range(len(denominators) - 1):
#                     preorder_list.append(', ')
#                     build_tree(denominators[i + 1], preorder_list)
#                     preorder_list.append(')')
#             else:
#                 build_tree(denominators[0], preorder_list)

#             preorder_list.append(')')

#         # 如果没有分母部分，正常处理为嵌套 Mul
#         else:
#             for _ in range(len(expr.args) - 1):
#                 preorder_list.append('Mul(')
#             build_tree(expr.args[0], preorder_list)
#             for i in range(len(expr.args) - 1):
#                 preorder_list.append(', ')
#                 build_tree(expr.args[i + 1], preorder_list)
#                 preorder_list.append(')')
#     elif expr.func.__name__ == 'Pow': #x**3 : ——> ['Mul(', 'Mul(', 'x', ', ', 'x', ')', ', ', 'x', ')']
#         # preorder_list.append('%s(' % expr.func.__name__)
#         # build_tree(expr.args[0], preorder_list)
#         # preorder_list.append(', ')
#         # build_tree(expr.args[1], preorder_list)
#         # preorder_list.append(')')

#         # we only deal with pow(x, y) whose y is integer
#         # print(expr.args[1])
#         # print(expr.args[0])
#         # print(expr.args[1])
#         # if isinstance(expr.args[0], ast.BinOp):  # 如果指数部分是一个二元运算（如乘法、加法等）
#         #     preorder_list.append('Pow(')
#         #     build_tree(expr.args[0], preorder_list)  # 先处理底数
#         #     preorder_list.append(', ')
#         #     build_tree(expr.args[1], preorder_list)  # 再处理指数部分（可能是包含变量的表达式）
#         #     preorder_list.append(')')
#         # if isinstance(expr.args[1], ast.BinOp):  # 如果指数部分是一个二元运算（如乘法、加法等）
#         #     print("kjljljljljljl")
#         #     preorder_list.append('Pow(')
#         #     build_tree(expr.args[0], preorder_list)  # 先处理底数
#         #     preorder_list.append(', ')
#         #     build_tree(expr.args[1], preorder_list)  # 再处理指数部分（可能是包含变量的表达式）
#         #     preorder_list.append(')')
#         # else:  # 如果指数部分是常数或者简单的数值
#         if re.match(r"^-?\d+(\.\d+)?$", str(expr.args[1])):
#             if float(expr.args[1]) % 1 == 0:
#                 if expr.args[1] < 0:
#                     preorder_list.append('Div(')
#                     preorder_list.append('1')
#                     preorder_list.append(', ')
#                     for _ in range(int(abs(expr.args[1])) - 1):
#                         preorder_list.append('Mul(')
#                     build_tree(expr.args[0], preorder_list)
#                     for i in range(int(abs(expr.args[1])) - 1):
#                         preorder_list.append(', ')
#                         build_tree(expr.args[0], preorder_list)
#                         preorder_list.append(')')
#                     preorder_list.append(')')
#                 elif expr.args[1] == 0:
#                     preorder_list.append('1')
#                 elif expr.args[1] > 0:
#                     for _ in range(int(expr.args[1]) - 1):
#                         preorder_list.append('Mul(')
#                     build_tree(expr.args[0], preorder_list)
#                     for i in range(int(expr.args[1]) - 1):
#                         preorder_list.append(', ')
#                         build_tree(expr.args[0], preorder_list)
#                         preorder_list.append(')')
#                 else:
#                     print('Unknown %s in Pow' % expr.args[1])
#                     exit(1)
#             else:
#                 preorder_list.append('Pow(')
#                 build_tree(expr.args[0], preorder_list)
#                 preorder_list.append(', ')
#                 build_tree(expr.args[1], preorder_list)  # 对指数部分也递归处理
#                 preorder_list.append(')')
#         else:
#             # preorder_list.append('Pow(')
#             preorder_list.append('Pow(%s)' % expr.args[0])
#             # build_tree(expr.args[0], preorder_list)  # 先处理底数
#             preorder_list.append(', ')
#             build_tree(expr.args[1], preorder_list)  # 再处理指数部分（可能是包含变量的表达式）
#             preorder_list.append(')')
#             # preorder_list.append('Pow(')
#             # build_tree(expr.args[0], preorder_list)
#             # preorder_list.append(', ')
#             # preorder_list.append('%s)' % expr.args[1])

#     elif expr.func.__name__ == 'exp' or expr.func.__name__ == 'sin' or expr.func.__name__ == 'cos' or expr.func.__name__ == 'tan' or expr.func.__name__ == 'log' or expr.func.__name__ == 'Abs':
#         preorder_list.append('%s(' % expr.func.__name__)
#         build_tree(expr.args[0], preorder_list)
#         preorder_list.append(')')
#     elif isinstance(expr, (sp.Symbol, sp.Number)):
#         preorder_list.append(str(expr))
#     elif expr is sp.zoo:
#         preorder_list.append(str(1.0))
#     elif expr.func.__name__ == 'Pi':
#         preorder_list.append(str(3.1415926))
#     elif expr.func.__name__ == 'I':
#         preorder_list.append(str(0))
#     elif expr.func.__name__ == 're':
#         preorder_list.append(str(1))
#     elif expr.func.__name__ == 'e':
#         preorder_list.append(str(2.718281828459))
#     elif expr.func.__name__ == 'floor':
#         preorder_list.append(str(1))
#     elif expr.func.__name__ == 'Exp1' or expr.func.__name__ == 'E':
#         preorder_list.append(str(2.718281828))
#     else:
#         print('Unknown %s' % expr.func.__name__)
#         exit(1)


def mut_Terminal(individual, pset, sampling_const):
    ephemerals_idx = [index
                      for index, node in enumerate(individual)
                      if isinstance(node, Terminal)]

    if len(ephemerals_idx) > 0:
        ephemerals_idx = random.choice(ephemerals_idx)
        # if not isinstance(individual[ephemerals_idx].value, str):
        #     constant
        individual[ephemerals_idx] = random.choice(
            [Terminal(sampling_const(), False, object), pset.terminals[object][0]])
        # individual[ephemerals_idx] = Terminal(sampling_const(), False, object)

    return individual


def mut_Operator(individual, pset):
    if len(individual) < 2:
        return individual

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:  # Terminal
        # do nothing
        return individual
    else:  # Primitive
        prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
        individual[index] = random.choice(prims)
    return individual


def mut_InsertNode(individual, pset):
    return mutInsert(individual, pset)[0]


def mut_SubtreeShrink(individual, pset):
    return mutShrink(individual)[0]


"""
An example of converter:
    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
        'cos': lambda x: sp.cos(x),
        'inv': lambda x: x ** (-1),
        'sqrt': lambda x: sp.sqrt(x),
    }
"""


def TreeToDeadStr(ind, converter):
    # return sp.sympify(str(ind), locals=converter)
    return sp.parse_expr(str(ind), local_dict=converter)


# def DeadStrToTree(read_str, converter):
#     return sp.sympify(str(ind), locals=converter)

def mut_Simplify(individual, pset, converter):
    eq_str_reduced = str(sp.simplify(TreeToDeadStr(individual, converter)))
    if eq_str_reduced is 'nan' or 'zoo' in eq_str_reduced:
        return individual
    else:
        return PrimitiveTree.from_string_sympy(eq_str_reduced, pset)


def mut_NewTree(individual, pset, min_, max_):
    return PrimitiveTree(genGrow(pset, min_, max_, type_=None))


def check_valid(expr):
    if len(expr.args) == 0:
        return True

    if expr.func.__name__ == 're' or expr.func.__name__ == 'im':
        return False

    if expr.func.__name__ == 'Pow':  # check the second args of Pow is number.
        flag1 = check_valid(expr.args[0])
        flag2 = check_valid(expr.args[1])
        try:
            float(expr.args[1])
        except:
            flag2 = False
        return flag1 and flag2
    elif expr.func.__name__ == 'exp' or expr.func.__name__ == 'sin' or expr.func.__name__ == 'cos' or expr.func.__name__ == 'tan' or expr.func.__name__ == 'log':
        if 'exp' in str(expr.args) or 'sin' in str(expr.args) or 'cos' in str(expr.args) or 'tan' in str(
                expr.args) or 'log' in str(expr.args):
            return False
        return check_valid(expr.args[0])
    else:
        flags = []
        for i in expr.args:
            flag_i = check_valid(i)
            flags.append(flag_i)
        return sum(flags) == len(flags)


def Mutations(individual, pset, sampling_const, converter, min_, max_, eval_func=None, x=None, y=None):
    individual_new = copy.deepcopy(individual)

    case = random.choice([1, 2, 3, 4, 5, 6])
    # print('case=', case)
    # case = 5
    if case == 1:
        # mutate constant or param
        individual_new = mut_Terminal(individual_new, pset, sampling_const)  # test ok
    elif case == 2:
        # mutate operator
        individual_new = mut_Operator(individual_new, pset)  # test ok
    elif case == 3:
        # mutate : insert node
        individual_new = mut_InsertNode(individual_new, pset)  # test ok
    elif case == 4:
        # mutate : shrink subtree
        individual_new = mut_SubtreeShrink(individual_new, pset)  # test ok
    elif case == 5:
        # mutate : simplify tree
        # individual_new = mut_Simplify(individual_new, pset, converter)  # too cost in time, so we ignore this mutation
        individual_new = individual_new
    elif case == 6:
        # mutate : new tree entirely
        individual_new = mut_NewTree(individual_new, pset, min_, max_)  # test ok
    else:
        print('Unknown case [%s] in Mutations!!!' % case)
        exit(1)

    if eval_func is None or x is None or y is None:
        return individual_new

    old_complex = len(individual)
    old_fitness = eval_func(individual, pset, x, y)

    new_complex = len(individual_new)
    new_fitness = eval_func(individual_new, pset, x, y)

    alpha = 0.1
    annealing_temperature = 1.0

    q_anneal = np.exp(-(new_fitness - old_fitness) / (alpha * annealing_temperature))
    q_parsimony = float(old_complex / new_complex)

    if random.uniform(0, 1) < q_anneal * q_parsimony:
        return individual_new
    else:
        return individual


def Mutations_f_g(individual_f_g, pset, sampling_const, converter, min_, max_, eval_func=None, x=None, y=None):
    pset_f_, pset_g_ = pset
    individual_f_new = Mutations(individual_f_g[0], pset_f_, sampling_const, converter, min_, max_, eval_func=eval_func,
                                 x=x, y=y)
    while not check_valid(TreeToDeadStr(individual_f_new, converter)):
        individual_f_new = Mutations(individual_f_g[0], pset_f_, sampling_const, converter, min_, max_,
                                     eval_func=eval_func,
                                     x=x, y=y)
    individual_g_new = Mutations(individual_f_g[1], pset_g_, sampling_const, converter, min_, max_, eval_func=eval_func,
                                 x=x, y=y)
    while not check_valid(TreeToDeadStr(individual_g_new, converter)):
        individual_g_new = Mutations(individual_f_g[1], pset_g_, sampling_const, converter, min_, max_,
                                     eval_func=eval_func,
                                     x=x, y=y)
    return (individual_f_new, individual_g_new)


