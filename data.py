import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    """支持 dense / sparse 输入的统一数据封装。

    中文总结:
    1. 沿用 scMRDR 系列的数据接口，保证训练脚本可以直接复用。
    2. 如果输入是稀疏矩阵，则按行取样时再转 dense，避免一开始整体 densify 占满内存。
    3. `w` 仍然保留在数据流里，便于和旧框架兼容，但 scMIGRA-v1 训练时不会使用 cell type 标签监督。
    """

    def __init__(self, X, b, m, i, w):
        super().__init__()
        self._x_is_sparse = issparse(X)
        if self._x_is_sparse:
            self.X = X.tocsr().astype(np.float32, copy=False)
        else:
            self.X = torch.tensor(X).float()

        self.len = X.shape[0]
        self.b = torch.tensor(b).float() if b is not None else torch.zeros(self.len).float()
        self.m = torch.tensor(m).float()
        self.i = torch.tensor(i).float()
        self.w = torch.tensor(w).float() if w is not None else torch.zeros(self.len).float()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self._x_is_sparse:
            x_row = torch.from_numpy(self.X[index].toarray().ravel()).float()
        else:
            x_row = self.X[index]
        return x_row, self.b[index], self.m[index], self.i[index], self.w[index]
