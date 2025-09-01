import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Default model path
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent.parent / "checkpoints" / "encoder_beta0.7.pt"

def get_ecfp4(smiles: str, n_bits: int = 2048):
    """
    Convert a SMILES string to an ECFP4 fingerprint tensor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return torch.tensor(fp, dtype=torch.float32)

def _load_model(model):
    """
    Load a model from path or return directly if already loaded.
    """
    import sys
    from dotP_encoder.model import CompoundEncoder
    sys.modules['__main__'].CompoundEncoder = CompoundEncoder
    if not torch.cuda.is_available():
        print("❌ GPU is not available. Running on CPU.")
        device='cpu'
    else:
        device='cuda'
    if isinstance(model, torch.nn.Module):
        return model
    elif isinstance(model, (str, Path)):
        model_path = Path(__file__).parent / "models" / model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        loaded_model = torch.load(model_path, map_location=device, weights_only=False)
        loaded_model.eval()
        return loaded_model
    else:
        raise ValueError("Model must be None, a path string, or a torch.nn.Module.")


def encoder(inputs, model=None, reduced=True):
    """
    Encode a list of SMILES strings or ECFP4 numpy arrays into dotP embeddings.

    Args:
        inputs (list[str] or list[np.ndarray]): SMILES strings or precomputed 2048-ECFP4 arrays.
        model (str or torch.nn.Module, optional): Path to a trained model or a loaded model. 
                                                  Defaults to _DEFAULT_MODEL_PATH.

    Returns:
        embeddings (torch.Tensor): Encoded vectors; zero vectors for invalid SMILES.
        valid_idx (list[int]): Indices of valid entries in the original list.
    """
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    fps = []
    valid_idx = []

    for idx, inp in enumerate(inputs):
        if isinstance(inp, str):
            fp = get_ecfp4(inp)
            if fp is not None:
                fps.append(fp)
                valid_idx.append(idx)
            else:
                print(f"[Warning] Invalid SMILES: {inp}")
        elif isinstance(inp, np.ndarray):
            fps.append(torch.tensor(inp, dtype=torch.float32))
            valid_idx.append(idx)
        else:
            raise TypeError(f"Input must be a SMILES string or a np.ndarray, got {type(inp)}")

    batch = torch.stack(fps)
    if reduced:
        # Capture intermediate output
        intermediates = {}
        def save_activation(name):
            def hook(module, input, output):
                intermediates[name] = output.detach().cpu()
            return hook

        handle = model.net[1].register_forward_hook(save_activation("dotP512"))
        with torch.no_grad():
            _ = model(batch)
        handle.remove()

        embeddings = intermediates["dotP512"]
    else:
        with torch.no_grad():
            embeddings = model(batch).cpu()

    return embeddings, np.array(valid_idx)

def encode_molecules(inputs, model=None, batch_size=1024, reduced=True):
    """
    Encode a list of SMILES strings or 2048-ECFP4 numpy arrays into dotP embeddings IN BATCHES.

    Args:
        inputs (list[str] or list[np.ndarray]): SMILES strings or precomputed 2048-ECFP4 arrays.
        model (str or torch.nn.Module, optional): Path to a trained model or a loaded model. 
                                                  Defaults to _DEFAULT_MODEL_PATH.

    Returns:
        embeddings (torch.Tensor): Encoded vectors; zero vectors for invalid SMILES.
        valid_idx (list[int]): Indices of valid entries in the original list.
    """
    batch_size = 1024 
    embeddings_list = []
    valid_idx_list = []

    if model is None:
        model = _DEFAULT_MODEL_PATH
    model = _load_model(model)

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i+batch_size]
        batch_embeddings, batch_valid_idx = encoder(batch_inputs, model=model, reduced=reduced)
        embeddings_list.append(batch_embeddings)
        valid_idx_list.extend(list(batch_valid_idx+i))

    return torch.cat(embeddings_list, dim=0).detach().cpu().numpy(), np.array(valid_idx_list)