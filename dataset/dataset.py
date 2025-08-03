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
    container = nib.Nifti1Image.get_fdata(img, dtype=np.float32)
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
    X = np.stack([mask_data_faster(img, mask) for img in imgs], dtype=np.float32)
    # print(X.dtype)
    return X  # subjects x time x voxels

def load_nii(mri_dir, mri_file):
    # """
    # Fallback loader for a single .nii file. Returns (header, tensor)
    # """
    # img = nib.load(path)
    # data = nib.Nifti1Image.get_fdata(img, dtype=np.float64)  # shape: time x X x Y x Z or similar
    # # flatten spatial dims to voxels (assuming first dim is time)
    # if data.ndim == 4:
    #     time = data.shape[0]
    #     voxels = np.prod(data.shape[1:])
    #     flat = data.reshape(time, voxels)
    # elif data.ndim == 3:
    #     # no time dimension: treat as single timepoint
    #     flat = data.reshape(1, -1)
    # else:
    #     raise ValueError(f"Unsupported nifti dimensionality: {data.shape}")
    # tensor = torch.tensor(flat, dtype=torch.float32)
    # return img.header, tensor  # header for compatibility, tensor is time x voxels
    mri_nii=nib.load(os.path.join(mri_dir, mri_file))
    mri=np.array(mri_nii.get_fdata(), dtype=np.float32)
    # 0-1 Normalization
    # mri=(mri-mri.min())/(mri.max()-mri.min())
    mri=torch.from_numpy(mri)
    return mri_nii, mri

class Dataset(data.Dataset):
    def __init__(self, data_in=None, num_modal=3, device='cpu', maskfname=None, w_reduce=None):
        super(Dataset, self).__init__()

        self.data_in = data_in
        self.num_modal = num_modal
        self.device = device
        self.maskfname = maskfname
        self.mat_data = []
        self.nii_data = []

        if isinstance(data_in, type(None)):
            self.data_dir=None
            self.data_files=None
        else:
            if isinstance(data_in, str) and os.path.isdir(data_in):
                self.data_dir=data_in
                self.data_files=os.listdir(data_in)
                self.data_files.sort()

            elif isinstance(data_in, str) and os.path.isfile(data_in):
                data_dir, data_file=os.path.split(data_in)
                self.data_dir=data_dir
                self.data_files=[data_file]

                if '.mat' in data_file:
                    try:
                        mat_file = sio.loadmat(self.data_in)
                    except:
                        mat_file = mat73.loadmat(self.data_in) # MATLAB -v7.3 usually for data > 2GB
                    # Expect 'X' key containing modal data
                    self.mat_data=[i.T for _, i in enumerate(np.squeeze(mat_file['X']))]
                    self.num_modal=len(self.mat_data)

                elif '.txt' in data_file:
                    if self.maskfname is None:
                        raise ValueError("maskfname must be provided when using a .txt subject list")
                    txt_path = os.path.join(self.data_dir, data_file) #if not os.path.isabs(self.data_in) else self.data_in
                    with open(txt_path, 'r') as f:
                        fnames = [line.strip() for line in f]#.readlines()]
                        fnames = [fname.replace("'", "").replace('"', '') for fname in fnames]

                    X = load_and_mask_nii(fnames, self.maskfname)
                    if w_reduce is not None:
                        W = np.load(w_reduce).astype(np.float32)
                        X = W[:X.shape[0]] @ X

                    self.nii_data = [torch.from_numpy(X[m].T) for m in range(X.shape[0])]
                    self.num_modal = len(self.nii_data)
                    
                    
                elif '.nii' in data_file or '.nii.gz' in data_file:
                    # Single .nii file input
                    _, tensor = load_nii(data_dir, data_file)
                    self.nii_data = [tensor]  
                    self.num_modal = 1

                else:
                    raise ValueError(f"Unsupported data_in type: {self.data_in}")
            else:
                print("Invalid data_in")
                sys.exit(1)


        # Normalize paths
        # data_file = os.path.basename(self.data_in)
        # data_dir = os.path.dirname(self.data_in)

        

    def __len__(self):
        if self.mat_data != []:
            # print(self.mat_data[0].shape)
            return self.mat_data[0].shape[0]
        elif self.nii_data != []:
            # print(self.nii_data[0].shape)
            return self.nii_data[0].shape[0]
        return len(self.data_files)


    def __getitem__(self, index):

        data_out=list()
        if self.mat_data == []:
            if self.nii_data == []:
                # list of .nii files 
                # (likely multimodal, one file per subject)
                _, mri=load_nii(self.data_dir, self.data_files[index])
                data_out.append(mri.to(self.device))
            else:
                for m in range(self.num_modal):
                    data_out.append(self.nii_data[m][index,:].to(self.device))
        else:
            # .mat file
            for i in range(self.num_modal):
                data_out.append(torch.tensor(self.mat_data[i][index,:], dtype=torch.float32, device=self.device))

        return data_out


if __name__ == '__main__':
    rootpath="/Users/xli77/Documents/MISA-pytorch/simulation_data"
    ds=Dataset(data_in=os.path.join(rootpath,"sim-siva.mat"))
    dl=data.DataLoader(dataset=ds, batch_size=1000, shuffle=True)
    for i, data_in in enumerate(dl):
        print(len(data_in), data_in[0].shape)
