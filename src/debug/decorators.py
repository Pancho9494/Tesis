from typing import Callable, Optional, List
import inspect
import torch
from itertools import count
import sys
import time
from dataclasses import dataclass, field

DEBUG = False


@dataclass
class CallStack:
    min_depth: int = field(default=9999)
    call_depths: list = field(default_factory=list)


min_depth = 9999
call_depths = []


def __current_indentation(callstack: CallStack) -> str:
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

    TAB = "\t"
    return TAB * callstack.call_depths.index(current_depth)


identify_method_callstack = CallStack()


def identify_method(after_msg: Optional[str] = None, on: bool = DEBUG):
    if isinstance(after_msg, Callable):
        # allow's the decorator to be called with the default argument without using parenthesis
        return identify_method()(after_msg)

    def decorator(method: Callable) -> Callable:
        def identify(self, *args, **kwargs) -> Callable:
            if not on:
                return method(self, *args, **kwargs)

            TABS = __current_indentation(identify_method_callstack)
            msg = f"{TABS}{self}.{method.__name__}(\n"
            signature = inspect.signature(method).bind(self, *args, **kwargs)
            signature.apply_defaults()

            for name, value in signature.arguments.items():
                if name == "self":
                    continue
                if isinstance(value, torch.Tensor):
                    msg += f"{TABS}\t{name}={value.shape}-> ({value.min()}, {value.max()}),\n"
                if isinstance(value, List):
                    if isinstance(value[0], torch.Tensor):
                        msg += f"{TABS}\t{name}=[{[v.shape for v in value]}]\n"
                else:
                    msg += f"{TABS}\t{name}={value},\n"
            msg += f"{TABS})"
            print(msg)
            ts = time.time()
            result = method(self, *args, **kwargs)
            te = time.time()
            print(f"{TABS}{chr(0x21AA)} {te - ts:2.6f} [s] ")

            if after_msg is not None:
                print(f"{TABS}{after_msg}")
            return result

        return identify

    return decorator


def timeit():
    def decorator(method: Callable) -> Callable:
        def time_call(self, *args, **kwargs):
            ts = time.time()
            result = method(self, *args, **kwargs)
            te = time.time()

            print(f"========================= {te - ts:2.6f} s =========================")
            return result

        return time_call

    return decorator
