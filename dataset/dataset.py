import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.utils.data as data
import mat73
import scipy.io as sio

def mask_data_faster(img, mask):
    """
    img: nibabel image
    mask: boolean numpy array (same spatial shape as img)
    Returns: T x V matrix (time x voxels selected by mask)
    """
    container = nib.Nifti1Image.get_fdata(img, dtype=np.float64)
    return container[mask].T  # T x V

def load_and_mask_nii(fnames, maskfname):
    """
    fnames: list of paths to .nii files (each assumed to be 4D: time x X x Y x Z)
    maskfname: path to a mask .nii where mask==1 indicates voxels to keep
    Returns: numpy array of shape (subjects, time, voxels)
    """
    # Load and binarize mask
    masker = nib.load(maskfname)
    mask = nib.Nifti1Image.get_fdata(masker, dtype=np.float32)
    mask = mask == 1  # boolean mask

    # Load each subject image and apply mask
    imgs = [nib.load(fname) for fname in fnames]
    # Each output is T x V; stack into (subjects x time x voxels)
    X = np.stack([mask_data_faster(img, mask) for img in imgs])
    return X  # subjects x time x voxels

def load_nii(path):
    """
    Fallback loader for a single .nii file. Returns (header, tensor)
    """
    img = nib.load(path)
    data = nib.Nifti1Image.get_fdata(img, dtype=np.float64)  # shape: time x X x Y x Z or similar
    # flatten spatial dims to voxels (assuming first dim is time)
    if data.ndim == 4:
        time = data.shape[0]
        voxels = np.prod(data.shape[1:])
        flat = data.reshape(time, voxels)
    elif data.ndim == 3:
        # no time dimension: treat as single timepoint
        flat = data.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported nifti dimensionality: {data.shape}")
    tensor = torch.tensor(flat, dtype=torch.float32)
    return img.header, tensor  # header for compatibility, tensor is time x voxels

class Dataset(data.Dataset):
    def __init__(self, data_in=None, num_modal=3, device='cpu', maskfname=None):
        super(Dataset, self).__init__()

        self.data_in = data_in
        self.num_modal = num_modal
        self.device = device
        self.maskfname = maskfname
        self.mat_data = []

        # Normalize paths
        data_file = os.path.basename(self.data_in)
        data_dir = os.path.dirname(self.data_in)

        if '.mat' in data_file:
            try:
                mat_file = sio.loadmat(self.data_in)
            except Exception:
                mat_file = mat73.loadmat(self.data_in)
            # Expect 'X' key containing modal data
            self.mat_data = [i.T for _, i in enumerate(np.squeeze(mat_file['X']))]
            self.num_modal = len(self.mat_data)

        elif '.txt' in data_file:
            if self.maskfname is None:
                raise ValueError("maskfname must be provided when using a .txt subject list")
            txt_path = os.path.join(data_dir, data_file) if not os.path.isabs(self.data_in) else self.data_in
            with open(txt_path, 'r') as f:
                fnames = [line.strip() for line in f.readlines()]

            X = load_and_mask_nii(fnames, self.maskfname)

            self.mat_data = [torch.tensor(X, dtype=torch.float32)]
            self.num_modal = len(self.mat_data)
            
            

        elif '.nii' in data_file or '.nii.gz' in data_file:
            # Single .nii file input
            header, tensor = load_nii(self.data_in)
            # Treat each modality as this tensor (could be expanded)
            self.mat_data = [tensor.numpy()]  # convert to numpy for consistency
            self.num_modal = 1

        else:
            raise ValueError(f"Unsupported data_in type: {self.data_in}")

    def __len__(self):
        # Length is number of subjects; for .mat, assume indexing along first dim of each modality
        if self.num_modal > 0:
            first = self.mat_data[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0]
            else:
                return first.shape[0]
        return 0

    def __getitem__(self, index):
        data_out = []
        # If mat_data entries are numpy, convert; if torch already, use directly
        for i in range(self.num_modal):
            entry = self.mat_data[i]
            if isinstance(entry, torch.Tensor):
                tensor_i = entry[index]
            else:
                tensor_i = torch.tensor(entry[index, :], dtype=torch.float32)
            # Move to device
            data_out.append(tensor_i.to(self.device))
        return data_out

if __name__ == '__main__':
    # Example usage with .mat file
    rootpath = "/Users/xli77/Documents/MISA-pytorch/simulation_data"
    ds = Dataset(data_in=os.path.join(rootpath, "sim-siva.mat"))
    dl = data.DataLoader(dataset=ds, batch_size=1000, shuffle=True)
    for i, data_in in enumerate(dl):
        print(len(data_in), data_in[0].shape)
