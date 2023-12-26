# 함수 실행 시간 알 수 있는 데코레이터 만들기
def logging_time(original_fn):
    import time
    from functools import wraps

    @wraps(original_fn)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)

        end_time = time.time()
        print("Running Time[{0}]: {1:0.8f} sec".format(original_fn.__name__, end_time - start_time))
        return result
    return wrapper