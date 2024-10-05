from torch.utils.data import Dataset
from pathlib import Path
from submodules.OverlapPredator.lib.benchmark import read_trajectory
import numpy as np
import concurrent.futures
from typing import List, Tuple
from database.pairs import Cloud, Pairs
import random

class ThreeDLoMatch(Dataset):
    """
    Class that represents the 3DLoMatch dataset
    
    """
    dir: Path
    pairs: List[Pairs]
    max_points: int = 30_000
    overlap_radius: float = 0.0375
    
    def __init__(self) -> None:
        self.dir = Path("database/raw_data/3DLoMatch")
        self.pairs = self._join_scene_ground_truths(shuffle=False)
        
        print(f"Loaded 3DLoMatch with {len(self)} point cloud pairs")
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Pairs: 
        pair = self.pairs[idx]
    
        if (len(pair.src) > self.max_points):
            pair.src.rand_downsample(self.max_points)
        if (len(pair.target) > self.max_points):
            pair.target.rand_downsample(self.max_points)
    
        return pair
        
    def _join_scene_ground_truths(self, shuffle: bool) -> List[Pairs]:
        allPairs = []
        for scene in (self.dir / "evaluation").iterdir():
            if not any(scene.iterdir()):
                print(f"No files found for scene {scene.stem}")
                continue
            try:
                keys, transforms = read_trajectory(self.dir / "evaluation" / scene.stem / "3dLoMatchGT.log")
                pair = [Pairs(
                            Cloud(self.dir / "fragments" / scene.stem / f"cloud_bin_{keys[idx][0]}.ply"),
                            Cloud(self.dir / "fragments" / scene.stem / f"cloud_bin_{keys[idx][1]}.ply"),
                            transforms[idx]
                        )
                        for idx in range(len(keys))]
                allPairs.extend(pair)
            except FileNotFoundError:
                print(f"Coudln't find 3dLoMatchGT.log file for scene {scene.stem}, run generate_3dLoMatch.py script")
                raise
        if shuffle:
               random.shuffle(allPairs)
        return allPairs
        
    def generate_lo_ground_truth(self) -> None:
        """
        Generates a new gt.log file that only has fragment pairs that aren't consecutive and that have less than 30%
        overlap

        """
        def _process_scene(scene: str) -> None:
            # print(f"Processing scene {scene}")
            keys, transforms = read_trajectory(self.dir / "evaluation" / scene / "gt.log")
            workload = [(self.dir / "fragments" / scene / f"cloud_bin_{keys[idx][0]}.ply", 
                        self.dir / "fragments" / scene / f"cloud_bin_{keys[idx][1]}.ply",
                        transforms[idx]) 
                for idx in range(len(keys))
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                these_futures = [executor.submit(compute_overlap, src, tgt, trans) for src, tgt, trans in workload]
                concurrent.futures.wait(these_futures)
            
            out = ""
            for pair, overlap in [future.result() for future in these_futures]:
                if ((pair.src.index + 1 == pair.target.index) or (overlap > 0.3)):
                    continue
                out += f"{pair.src.index}\t {pair.target.index}\t {keys[0][-1]}	\n{np_to_str(pair.truth)}"
            with open(self.dir / "evaluation" / scene / "3dLoMatchGT.log", "w") as log_file:
                log_file.write(out)
                
        for scene in (self.dir / "evaluation").iterdir():
            dir_empty = not any(scene.iterdir())
            if dir_empty:
                print(f"No files found for scene {scene.stem}")
                continue
            _process_scene(scene.stem)

def compute_overlap(src: Path, target: Path, transform: np.ndarray) -> Tuple[Pairs, float]:
    """
    Utility function that computes the overlap of two fragments, given their ground truth transformation

    Args:
        src (Path): Path to the source fragment
        target (Path): Path to the target fragment
        T (np.ndarray): 4x4 Transformation matrix that aligns target to src

    Returns:
        FragmentPairs: _description_
    """
    pair = Pairs(Cloud(src), Cloud(target), transform)
    overlap = pair.GT_overlap()
    # pair.compute_overlap() # TODO: if I call this futures complains that it cannot pickle o3d Point Clouds
    return pair, overlap

        
def np_to_str(arr: np.ndarray) -> str:
    """
    Uitlity function, converts a multidimensional numpy array to the appropiate format for the .log files

    Args:
        arr (np.ndarray): The array being converted

    Returns:
        str: The string representation of the array
    """
    out = ""
    tab = '\t'
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            out += f"{'' if arr[col, row] < 0 else ' '}{arr[col, row]:.8e}{tab + ' ' if row < arr.shape[0]-1 else tab}"
        out += "\n"
    return out