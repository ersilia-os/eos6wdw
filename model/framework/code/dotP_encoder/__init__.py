"""
dotp_encoder
------------

Encode molecules into dotP embeddings using a pretrained CompoundEncoder.

Example:
    import dotp_encoder as de
    smiles = ["CCO", "CCN", "invalid"]
    embeddings, valid_idx = de.encode_smiles_list(smiles)
"""

from .model import CompoundEncoder
from .encode_smiles import get_ecfp4, encoder, encode_molecules

__all__ = ["CompoundEncoder", "get_ecfp4", "encode_molecules", 'encoder']
