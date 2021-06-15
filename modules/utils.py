import numpy as np
def progress_print(to_print: str, print_flag: bool):
    if print_flag:
        print(to_print)
    else:
        pass

def f_symlog(x):
    if -1 > x > 1:
        return x
    elif x>1:
        return np.log(x)
    elif x<-1:
        return -np.log(-x)