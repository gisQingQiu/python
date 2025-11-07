
from functools import wraps
import datetime

def timeit(func):
    '''
    装饰器，计算函数运行时间
    
    Parameters
    -----------------------------------------------
    func
        需要装饰的函数
        
    Returns:
    -----------------------------------------------
    wrapper
    '''
    
    @wraps(func)    # 装饰器
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()           # 记录开始时间
        result = func(*args, **kwargs)                 # 执行原函数
        end_time = datetime.datetime.now()             # 记录结束时间
        print(f"函数 {func.__name__} 运行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper





