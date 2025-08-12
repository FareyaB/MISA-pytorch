
import os
import numpy as np
import mat73

iva_paths = ['/data/users3/rsilva/MISA-pytorch/results/2025bhi-rebuttal/iva_gl_default.mat',
             '/data/users3/rsilva/MISA-pytorch/results/2025bhi-rebuttal/iva_gl_default_no_whiten.mat',
             '/data/users3/rsilva/MISA-pytorch/results/2025bhi-rebuttal/iva_g_default.mat',
             '/data/users3/rsilva/MISA-pytorch/results/2025bhi-rebuttal/iva_gl_eye.mat',
             '/data/users3/rsilva/MISA-pytorch/results/2025bhi-rebuttal/iva_g_eye.mat']

gpca_input_full_path = "/data/users3/rsilva/dask-iva/W_gpca_M100_K12_ukb_fMRI.npy"

Wgpca = np.load(gpca_input_full_path)

for iva_input_full_path in iva_paths:

    Wiva = mat73.loadmat(iva_input_full_path)['W']
    # retrieve matrices from list object:
    Wiva = np.stack([np.array(i[0]) for i in Wiva])
    W = Wiva @ Wgpca

    gpca_filename = os.path.split(gpca_input_full_path)[1]

    dirname, iva_filename = os.path.split(iva_input_full_path)
    fname = f"W_{iva_filename.split('.')[0]}_{gpca_filename}"

    print(f"Saving W matrix to {os.path.join(dirname, fname)}")

    # np.save(os.path.join(dirname, fname), W)
# print(W.shape)
# print(Wiva[0][0][0][0]) 
# print(len(Wiva))
# print(len(Wiva[0]))
# print(len(Wiva[0][0]))
# print(len(Wiva[0][0][0]))
# print(W[0,:3,:4])
# print(W[1,:3,:4])


