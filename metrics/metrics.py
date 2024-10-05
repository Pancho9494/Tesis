from database.pairs import Pairs
from torch.utils.data import Dataset
from metrics.metricI import PairMetric, DatasetMetric
import copy
import open3d as o3d
import numpy as np

class RootMeanSquaredError(PairMetric):
    def __call__(self, pair: Pairs) -> float:
        pred, truth = copy.deepcopy(pair.target), copy.deepcopy(pair.target)
        pred_arr = np.asfarray(pred.pcd.transform(pair.prediction).points)
        truth_arr = np.asfarray(truth.pcd.transform(pair.truth).points)
        return np.sqrt(((pred_arr - truth_arr) ** 2).mean())
        

class RegistrationRecall(DatasetMetric):
    """
    The fraction of scan pairs for which the correct transformation parameters are found with RANSAC i.e. the RMSE is 
    smaller than 0.2
    """
    
    def __call__(self, data: Dataset) -> float:
        RMSE = RootMeanSquaredError()
        recall = 0.0
        for pair in data:
            rmse = RMSE(pair)
            print(f"{pair} has RMSE of {rmse}")
            if rmse < 0.2:
                recall += 1
        return recall / len(data)
        
class FeatureMatchRecall(PairMetric):
    """
    The fraction of pairs that have >5% inlier matches with <10 cm residual under the ground truth transformation
    """
    
    def __call__(self, pair: Pairs) -> float:
        ...
        
class InlierRatio(PairMetric):
    """
    The fraction of correct correspondences among the putative matches
    """
    
    def __call__(self, pair: Pairs) -> float:
        ...
