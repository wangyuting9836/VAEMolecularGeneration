import torch
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

from data_utils import indices_to_smiles


@torch.no_grad()
def generate_from_latent(model, z, idx2char):
    model.eval()
    logits = model.decode(z.unsqueeze(0))[0]  # (T, V)
    indices = logits.argmax(dim=-1)  # (T,)
    return indices_to_smiles(indices.cpu(), idx2char)


@torch.no_grad()
def generate_from_latent(model, z, idx2char):
    model.eval()
    logits = model.decode(z.unsqueeze(0))[0]  # (T, V)
    indices = logits.argmax(dim=-1)  # (T,)
    return indices_to_smiles(indices.cpu(), idx2char)


def interpolate(model, z1, z2, n_steps=10, **kwargs):
    alphas = np.linspace(0, 1, n_steps)
    smiles_list = []
    for alpha in alphas:
        z = alpha * z2 + (1 - alpha) * z1
        smi = generate_from_latent(model, z.unsqueeze(0), **kwargs)
        smiles_list.append(smi)
    return smiles_list


def visualize_one_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol:
        # Draw.MolToFile(mol, "molecule.png", size=(300, 300), kekulize=True)
        Draw.MolToImage(mol, size=(300, 300), kekulize=True)
        Draw.ShowMol(mol, size=(300, 300), kekulize=False)
    else:
        print("Invalid molecule to display.")


def visualize_smiles(smiles_list, mols_per_row=5):
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row)
        img.show()
    else:
        print("No valid molecules to display.")
