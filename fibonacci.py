import time
from functools import lru_cache

def fibonacci_recursive(n: int) -> int:
    """再帰実装
    
    引数:
        n (int): 計算するフィボナッチ数列の項数
    
    戻り値:
        int: n 番目のフィボナッチ数
    
    例外:
        ValueError: n が負の整数の場合
    """
    if n < 0:
        raise ValueError("n は 0 以上の整数である必要があります")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

@lru_cache(maxsize=None)
def fibonacci_memoized(n: int) -> int:
    """メモ化実装
    
    引数:
        n (int): 計算するフィボナッチ数列の項数
    
    戻り値:
        int: n 番目のフィボナッチ数
    
    例外:
        ValueError: n が負の整数の場合
    """
    if n < 0:
        raise ValueError("n は 0 以上の整数である必要があります")
    if n <= 1:
        return n
    return fibonacci_memoized(n - 1) + fibonacci_memoized(n - 2)

def fibonacci_loop(n: int) -> int:
    """ループ実装
    
    引数:
        n (int): 計算するフィボナッチ数列の項数
    
    戻り値:
        int: n 番目のフィボナッチ数
    
    例外:
        ValueError: n が負の整数の場合
    """
    if n < 0:
        raise ValueError("n は 0 以上の整数である必要があります")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def compare_speeds(n: int = 35) -> None:
    """速度比較
    
    引数:
        n (int): 計算するフィボナッチ数列の項数（デフォルト: 35）
    
    戻り値:
        None
    """
    methods = [
        ("再帰", fibonacci_recursive),
        ("メモ化", fibonacci_memoized),
        ("ループ", fibonacci_loop)
    ]
    
    print(f"n = {n} のフィボナッチ数列速度比較")
    print("-" * 50)
    
    for name, func in methods:
        start = time.perf_counter()
        result = func(n)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{name}: 結果={result}, 時間={elapsed:.6f}秒")
    
    print("-" * 50)

if __name__ == "__main__":
    compare_speeds()
