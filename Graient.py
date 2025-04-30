import numpy as np
from typing import List, Dict, Any

#梯度计算函数
def gradient(x: np.ndarray,target ,subset: np.ndarray = None ) -> np.ndarray:
    # 示例梯度计算（假设为线性回归）
    if subset is None:
        subset = subset
    X, y = subset, target
    return 2 * (X.T @ (X @ x - y)) / len(subset)