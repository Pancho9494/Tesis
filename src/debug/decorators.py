from typing import Callable, Optional
import inspect
import torch
from itertools import count
import sys

min_depth = 9999
call_depths = []

def identify_method(after_msg: Optional[str] = None):
    if isinstance(after_msg, Callable):
        # allow's the decorator to be called with the default argument without using parenthesis
        return identify_method()(after_msg)
    
    def decorator(method: Callable) -> Callable:
        def identify(self, *args, **kwargs) -> Callable:
            global min_depth, call_depths
            if (current_depth := __stack_size3a()) < min_depth:
                min_depth = current_depth
            if current_depth not in call_depths:
                call_depths.append(current_depth)
            
            TABS = "\t" * call_depths.index(current_depth)
            msg = f"{TABS}{self}.{method.__name__}(\n"
            signature = inspect.signature(method).bind(self, *args, **kwargs)
            signature.apply_defaults()
        
            for name, value in signature.arguments.items():
                if name == "self":
                    continue
                if isinstance(value, torch.Tensor):
                    msg += f"{TABS}\t{name}={value.shape}-> ({value.min()}, {value.max()}),\n"
                else:
                    msg += f"{TABS}\t{name}={value},\n"
            msg += f"{TABS})"
            print(msg)
            result = method(self, *args, **kwargs)
            
            if after_msg is not None:
                print(f"{TABS}{after_msg}")
            return result
        return identify
    
    return decorator





def __stack_size3a(size=2):
    """Get stack size for caller's frame.
    """
    frame = sys._getframe(size)
    try:
        for size in count(size, 8):
            frame = frame.f_back.f_back.f_back.f_back.\
                f_back.f_back.f_back.f_back
    except AttributeError:
        while frame:
            frame = frame.f_back
            size += 1
        return size - 1
