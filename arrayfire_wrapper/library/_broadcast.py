from collections.abc import Callable
from typing import Any


class Bcast:
    def __init__(self) -> None:
        self._flag: bool = False

    def get(self) -> bool:
        return self._flag

    def set(self, flag: bool) -> None:
        self._flag = flag

    def toggle(self) -> None:
        self._flag ^= True


bcast_var: Bcast = Bcast()

# TODO Remove
# def broadcast(func: Callable[..., Any], *args: Any) -> Any:
#     """
#     Function to perform broadcast operations.

#     This function can be used directly or as an annotation in the following manner.

#     Example
#     -------

#     Using broadcast as an annotation

#     >>> import arrayfire as af
#     >>> @af.broadcast
#     ... def add(a, b):
#     ...     return a + b
#     ...
#     >>> a = af.randu(2,3)
#     >>> b = af.randu(2,1) # b is a different size
#     >>> # Trying to add arrays of different sizes raises an exceptions
#     >>> c = add(a, b) # This call does not raise an exception because of the annotation
#     >>> af.display(a)
#     [2 3 1 1]
#         0.4107     0.9518     0.4198
#         0.8224     0.1794     0.0081

#     >>> af.display(b)
#     [2 1 1 1]
#         0.7269
#         0.7104

#     >>> af.display(c)
#     [2 3 1 1]
#         1.1377     1.6787     1.1467
#         1.5328     0.8898     0.7185

#     Using broadcast as function

#     >>> import arrayfire as af
#     >>> add = lambda a,b: a + b
#     >>> a = af.randu(2,3)
#     >>> b = af.randu(2,1) # b is a different size
#     >>> # Trying to add arrays of different sizes raises an exceptions
#     >>> c = af.broadcast(add, a, b) # This call does not raise an exception
#     >>> af.display(a)
#     [2 3 1 1]
#         0.4107     0.9518     0.4198
#         0.8224     0.1794     0.0081

#     >>> af.display(b)
#     [2 1 1 1]
#         0.7269
#         0.7104

#     >>> af.display(c)
#     [2 3 1 1]
#         1.1377     1.6787     1.1467
#         1.5328     0.8898     0.7185

#     """

#     def wrapper(*func_args: Any) -> Any:
#         bcast_var.toggle()
#         res = func(*func_args)
#         bcast_var.toggle()
#         return res

#     if len(args) == 0:
#         return wrapper()
#     else:
#         return wrapper(*args)
