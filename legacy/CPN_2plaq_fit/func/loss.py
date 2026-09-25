# from torch import nn
# import torch

# class ContNLLLoss(nn.Module):
#     """
#     输入概率密度，输出连续负对数似然损失

#     :param reduction: 'mean', 'sum', or 'none'，指定如何聚合损失值

#      - 'mean': 返回所有样本的平均损失

#      - 'sum': 返回所有样本的损失之和

#      - 'none': 返回每个样本的损失值，不进行聚合 
#     """
#     def __init__(self, reduction='mean'):
#         super().__init__()
#         self.reduction = reduction
    
#     def forward(self, probs):
#         """
#         probs: 概率密度，形状 (N,) 或 (N, 1)
#         """
#         log_probs = torch.log(probs)
#         if self.reduction == 'mean':
#             return -log_probs.mean()
#         elif self.reduction == 'sum':
#             return -log_probs.sum()
#         elif self.reduction == 'none':
#             return -log_probs
#         else:
#             raise ValueError(f"Invalid reduction: {self.reduction}")