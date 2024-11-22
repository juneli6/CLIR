import numpy as np

class NDCG():
    def __init__(self) -> None:
        pass
    
    def dcg_at_k(self, scores, k):
        dcg = 0
        for i, score in enumerate(scores[:k]):
            dcg += np.exp2(score) / np.log2(i + 2)
        return dcg
    
    def idcg_at_k(self, targets, k):
        idcg = 0
        for i, target in enumerate(targets[:k]):
            idcg += np.exp2(target[1]) / np.log2(i + 2)
        return idcg
    
    def ndcg_at_k(self, preds, targets, k):
        idcg = self.idcg_at_k(targets, k)
        dcg = self.dcg_at_k(preds, k)
        return dcg / idcg