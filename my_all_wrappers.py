from functools import wraps

def memorize(func):
    """Store the results of the decorated function for fast lookup
    """
    # Store results in a dict that maps arguments to results
    cache = {}
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # If these arguments haven't been seen before,
        params=tuple(args)+ tuple(kwargs)
        if params not in cache:
            # Call func() and store the result.
            cache[params] = func(*args, **kwargs)
        return cache[params]
    return wrapper

# if __name__=="__main__":
#     import time

#     @memorize
#     def slow_add(a, b):
#         time.sleep(5)
#         return a + b

#     #takes 5 second the first time
#     print(slow_add(1,2))
#     #immediately printed
#     print(slow_add(1,2))


def counter(func):
    """func.count returns the number of times a func is executed"""
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        # Call the function being decorated and return the result
        return func(*args, **kwargs)
    wrapper.count = 0
    # Return the new decorated function
    return wrapper

import time
def timer(func):
    """A decorator that prints how long a function took to run."""
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result.
        result = func(*args, **kwargs)
        # Get the total time it took to run, and print it.
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper

# if __name__=="__main__":
#     import numpy as np
#     @timer
#     def get_random(n):
#         rand_nums = np.random.rand(n)
#         return rand_nums

#     get_random(10000000)


from threading import Thread
from functools import wraps

def timeout(timeout):
    """A decorator that rises an Exception if a function took longer than the given seconds to run."""
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('TimeoutError: function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

# if __name__=="__main__":
#     import time

#     @timeout(2)
#     def foo(a,b):
#         time.sleep(1)
#         print('foo!')
#         return a+b

#     print(foo(2,3))

def tag(*tags):
  # Define a new decorator, named "decorator", to return
  def decorator(func):
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
      # Call the function being decorated and return the result
      return func(*args, **kwargs)
    wrapper.tags = tags
    return wrapper
  # Return the new decorator
  return decorator

# if __name__=="__main__":

#     @tag('test', 'this is a tag')
#     def foo():
#         pass
#     print(foo.tags)

import pprint
def wrap_result(start,end="",pretty_print=False):
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            s=func(*args, **kwargs)
            if pretty_print:
                s= pprint.pformat(s) 

            prefix=f'\n\n--------{start}--------\n'
            
            if end=="":
                suffix='\n' + '-'*(len(prefix)-3) + '\n'
            else:
                suffix=f'\n--------{end}--------\n'
                # i=(len(prefix)-len(end))//2
                # suffix='-'*
            return f'{prefix}{s}{suffix}'    
            
        return wrapper
    # Return the new decorator
    return decorator
# if __name__=="__main__":
#     @wrap_result("top unqiue words in the case as opposed to other cases")
#     def f():
#         return {'Alex Green Elementary': 0.030287172719682773, 'Bellshire Elementary': 0.0988045140909651,
# 'Brick Church College Prep': 0.08961013862715599, 'Buena Vista Elementary': 0.10570511270825833,
# 'Cockrill Elementary': 0.1077685612196105}
#     s=f()
#     print(s)

# def double_args(func):
#     def wrapper(a,b):
#         return func(a,b)
#     return wrapper


