import inspect
import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from itertools import count
from typing import Callable, Iterable, List, Optional

import torch

DEBUG = False


@dataclass
class CallStack:
    min_depth: int = field(default=9999)
    call_depths: list = field(default_factory=list)


min_depth = 9999
call_depths = []


def __current_indentation(callstack: CallStack) -> int:
    def __stack_size3a(size=2):
        """Get stack size for caller's frame."""
        frame = sys._getframe(size)
        try:
            for size in count(size, 8):
                frame = frame.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back
        except AttributeError:
            while frame:
                frame = frame.f_back
                size += 1
            return size - 1

    if (current_depth := __stack_size3a()) < callstack.min_depth:
        callstack.min_depth = current_depth
    if current_depth not in callstack.call_depths:
        callstack.call_depths.append(current_depth)

    return callstack.call_depths.index(current_depth)


identify_method_callstack = CallStack()

nested_colors: List[str] = ["green", "yellow", "blue", "red", "cyan"]


def identify_method(after_msg: Optional[str] = None, on: bool = DEBUG):
    if isinstance(after_msg, Callable):
        # allow's the decorator to be called with the default argument without using parenthesis
        return identify_method()(after_msg)

    def decorator(method: Callable) -> Callable:
        def identify(self, *args, **kwargs) -> Callable:
            if not on:
                return method(self, *args, **kwargs)

            depth = __current_indentation(identify_method_callstack)
            TABS = "    " * depth
            color = nested_colors[depth % len(nested_colors)]

            header = f"{TABS}[bold {color}]{self}[/bold {color}].[italic {color}]{method.__name__}([/italic {color}]"
            print(header)

            signature = inspect.signature(method).bind(self, *args, **kwargs)
            signature.apply_defaults()

            for name, value in signature.arguments.items():
                if name == "self":
                    continue
                body = f"{TABS}\t[{color}]{name}[/{color}]="
                if isinstance(value, torch.Tensor):
                    body += f"{value.shape}-> ({value.min()}, {value.max()}),"
                elif isinstance(value, List) and isinstance(value[0], torch.Tensor):
                    body += f"[{[v.shape for v in value]}]"
                else:
                    body += f"{value},"
                print(body)

            ts = time.time()
            result = method(self, *args, **kwargs)
            te = time.time()

            footer = f"{TABS}[italic {color}])[/italic {color}] -> {te - ts:2.6f} [cyan]\[s][/cyan] "
            print(footer)

            if after_msg is not None:
                print(f"{TABS}{after_msg}")
            return result

        return identify

    return decorator


def only_kwargs(arg: Callable | Iterable[str] | None = None, /, *, expected_args: Iterable[str] | None = None):
    """
    Decorator that rejects positional args (beyond `self`).
    - Use as `@only_kwargs` (no validation)
    - Or     `@only_kwargs(["R","center","idx"])`  (list as first arg)
    - Or     `@only_kwargs(expected_args=["R","center"])` (kw form)

    If `expected_args` is provided, it will also error on unexpected kwargs.
    """
    # Normalize the two calling styles
    if callable(arg):
        func = arg
        exp = None
    else:
        func = None
        exp = set(arg) if arg is not None else (set(expected_args) if expected_args is not None else None)

    def decorator(f: Callable):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if args:
                msg = f"{self.__class__.__name__}.{f.__name__} only accepts keyword arguments" + (
                    f"; allowed keys: {sorted(exp)}" if exp is not None else ""
                )
                raise ValueError(msg)
            if exp is not None:
                extra = set(kwargs) - exp
                if extra:
                    raise TypeError(
                        f"Unexpected keyword(s) for {self.__class__.__name__}.{f.__name__}: "
                        f"{sorted(extra)}. Allowed: {sorted(exp)}"
                    )
            return f(self, **kwargs)

        return wrapper

    return decorator if func is None else decorator(func)
