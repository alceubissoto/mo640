import os
from skbio import DistanceMatrix
from skbio.tree import nj
import numpy as np

directory_path = '../data'
directory = os.fsencode(directory_path)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("additive.npy"):
        data = np.load(os.path.join(directory_path, filename))
        dm = DistanceMatrix(data)
        nj_tree = nj(dm)
        df = nj_tree.tip_tip_distances().to_data_frame()

        df.index = df.index.astype(int)
        df.sort_index(inplace=True)
        df.columns = df.columns.values.astype(np.int32)
        df = df[sorted(df.columns)]

        print('source original dist matrix')
        print(data)
        print('calculated dist matrix')
        print(df)

        assert(data, df.astype(np.int32))