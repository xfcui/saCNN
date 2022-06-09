import argparse
import h5py
from Preprocessing import Preprocessing as P
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prot_filename", type=str, default="", help="protein data file name")
    parser.add_argument("--lig_filename", type=str, default="", help="ligand file name")
    parser.add_argument("--feature_filename", type=str, default="", help="output file name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    data = P.calcDatasetVoxel(P(boxsize=[24, 24, 24]), protPath=args.prot_filename, ligPath=args.lig_filename)
    if data:
        dataset, c, feature_protein, feature_ligand, feature_protein_shaped, feature_ligand_shaped = data
        with h5py.File(args.feature_filename, 'w') as f:
            f.create_dataset('data', data=np.array(dataset))
        print(f'The characterization of the complex with {args.prot_filename} is completed')
    else:
        print(f'The characterization of the complex with {args.lig_filename} is fail')


