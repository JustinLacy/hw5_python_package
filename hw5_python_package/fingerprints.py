from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

def smiles_to_maccs(smiles_list: List[str]) -> np.ndarray:
    fps = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:
            mol = None
        if mol is None:
            fps.append(np.zeros(166, dtype=int))
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)  # ExplicitBitVect with 166 bits
        arr = np.array([int(fp[i]) for i in range(166)], dtype=int)
        fps.append(arr)
    return np.vstack(fps)

def smiles_to_morgan(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    fps = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:
            mol = None
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=int))
            continue
        bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.array([int(bitvec[i]) for i in range(n_bits)], dtype=int)
        fps.append(arr)
    return np.vstack(fps)
