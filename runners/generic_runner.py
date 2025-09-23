import os
import torch
import mat73
import numpy as np
import scipy.io as sio
import pickle
from dataset.dataset import Dataset
from torch.utils.data import DataLoader
from scipy.stats import loguniform
from model.misa_wrapper import MISA_wrapper

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

def zscore(data, axis):
    data -= data.mean(axis=axis, keepdims=True)
    data /= data.std(axis=axis, keepdims=True)
    return np.nan_to_num(data, copy=False)

def correlation(matrix1, matrix2):

    # 1/stdev(M1,axis=1)[None,:] * (M1 @ M2.T) * 1/stdev(M2,axis=1)[:,None] / (d1-1)

    d1 = matrix1.shape[-1]
    d2 = matrix2.shape[-1]

    assert d1 == d2
    assert matrix1.ndim <= 2
    assert matrix2.ndim <= 2
    
    matrix1 = zscore(matrix1.astype(float), matrix1.ndim - 1) / np.sqrt(d1)
    matrix2 = zscore(matrix2.astype(float), matrix2.ndim - 1) / np.sqrt(d2)

    if matrix1.ndim >= matrix2.ndim:
        return matrix1 @ matrix2.T
    else:
        return matrix2 @ matrix1.T

def run_misa(args, config):
    """run MISA"""

    # From args:
    data = args.data.lower()
    data_filename = args.filename
    w = args.weights
    initial_weights = []

    test = args.test
    A_exist = args.a_exist
    
    # From config:
    device = config.device # this is set in main.py

    if 'mask_name' in config:
        mask_name = config.mask_name
        if mask_name.lower() in ['simtb16']:
            data_seed = config.data_seed
        # elif mask_name.lower() in ['ukb2907-smri-aal2']:
    else:
        mask_name = None

    if 'input_dim' in config:
        input_dim = config.input_dim

    if 'output_dim' in config:
        output_dim = config.output_dim
    elif 'input_dim' in config:
        output_dim = config.input_dim

    if 'subspace' in config:
        subspace = config.subspace
    else:
        pass # TO DO: should error...

    if 'eta' in config:
        eta = config.eta
    
    if 'beta' in config:
        beta = config.beta

    if 'lam' in config:
        lam = config.lam

    if config.special.nRuns != []:
        nRuns = config.special.nRuns
    else:
        nRuns = loguniform_int(0, 10).rvs(size=1)[0]
    
    if config.special.epochs != []:
        epochs = config.special.epochs
    else:
        # the integer type returned by loguniform_int is int64, 
        # which can't recognized as an int in DataLoader,
        # so need to cast int64 to int here
        epochs = int(loguniform_int(100, 500).rvs(size=1)[0])

    if config.special.batch_size != []:
        batch_size = config.special.batch_size
    else:
        batch_size = int(loguniform_int(20, 1000).rvs(size=1)[0])
    
    if config.special.lr != []:
        lr = config.special.lr
    else:
        lr = loguniform.rvs(0.00001, 0.1, size=1)[0]
    
    if config.special.adam_betas != []:
        adam_betas = config.special.adam_betas

    if data.lower() == 'mat':
        # load the data
        # matfile = os.path.join('./simulation_data', 'sim-{}.mat'.format(config.dataset))
        datafile = os.path.join('./simulation_data', data_filename)
    else:
        datafile = data_filename
    
    # LOAD GROUND-TRUTH A MATRIX
    ground_truth_A = None
        
    if data.lower() == 'mat':
        # load ground-truth sources for comparison
        # s = ...
        try:
            matdict=sio.loadmat(datafile)
            W=np.squeeze(matdict[w])
            if A_exist:
                matA=np.squeeze(matdict['A'])
                matY=np.squeeze(matdict['Y'])
        except:
            matdict=mat73.loadmat(datafile)
            W=matdict[w]
            if A_exist:
                matA=matdict['A']
                matY=matdict['Y']
        
        # LOAD INITIAL WEIGHTS
        initial_weights = [i for _, i in enumerate(W)]
        
        if A_exist:
            ground_truth_A = [i for _, i in enumerate(matA)]
            ground_truth_Y = [i.T for _, i in enumerate(matY)]

    else:
        if mask_name.lower() in ['simtb16']:
            pass
        elif mask_name.lower() in ['ukb2907-smri-aal2']:
            pass

    if initial_weights == []:
        ds = Dataset(data_in=datafile, device=device, maskfname=mask_name, w_reduce=w)
        initial_weights = [np.eye(dd.shape[-1]) for dd in ds.nii_data]
        # print("Initial weights (NIfTI):", [w.shape for w in initial_weights])
        # print(ds.num_modal, "modalities found in the dataset.")
    else:
        ds=Dataset(data_in=datafile, device=device, maskfname=mask_name)
    if len(ds) < batch_size:
        batch_size = len(ds)
    train_data200=DataLoader(dataset=ds, batch_size=200, shuffle=True)
    train_data300=DataLoader(dataset=ds, batch_size=300, shuffle=True)
    test_data=DataLoader(dataset=ds, batch_size=len(ds), shuffle=False)

    num_modal = ds.num_modal
    index = slice(0, num_modal)
        
    if ds.mat_data != []:
        input_dim = [torch.tensor(dd.shape[-1],device=device) for dd in ds.mat_data]

    if ds.nii_data != []:
        input_dim = [torch.tensor(dd.shape[-1],device=device) for dd in ds.nii_data]

    # if config.output_dim != []:
    #     output_dim = [torch.tensor(dd,device=device) for dd in output_dim]
    # else:
    output_dim = input_dim

    # TODO: make this an option to set the initial weights rather than default to data reduction and initilize with identity matrix
    if initial_weights == []:
        if isinstance(w, str) and os.path.isfile(w):
            if w.endswith('.pkl'):
                with open(w, 'rb') as f:
                    W = pickle.load(f)
            elif w.endswith('.npy') or w.endswith('.npz'):
                W = np.load(w)
                # if isinstance(W, np.lib.npyio.NpzFile):  # Handle .npz
                #     W = W['arr_0']  # default key
            else:
                raise ValueError("Unsupported weight file format: {}".format(w))

            # Conversion to torch Tensor happens in MISAK.py init
            initial_weights = [W[m] for m in range(W.shape[0])]
        # else:
        #     initial_weights = w
        

    # if subspace is a string, convert it to a list of tensors
    if isinstance(subspace, str):
        if subspace.lower() == 'iva':
            subspace_name = 'iva'
            subspace = [torch.eye(dd, device=device) for dd in output_dim]
    else:
        # check if it's a list of integers
        if isinstance(subspace, list) and all(isinstance(i, int) for i in subspace):
            # check if all elements are 1:
            if all(i == 1 for i in subspace):
                subspace_name = 'iva'
                subspace = [torch.eye(dd, device=device) for dd in output_dim]
            else:
                subspace_name = 'custom'
                # subspace = [torch.eye(dd, device=device) for dd in subspace]
        else:
            raise ValueError("Subspace must be a string or a list of integers")
        
    if len(eta) > 0:
        eta = torch.tensor(eta, dtype=torch.float32, device=device)
        if len(eta) == 1:
            eta = eta*torch.ones(subspace[0].size(-2), device=device)
    else:
        # should error
        pass

    if len(beta) > 0:
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        if len(beta) == 1:
            beta = beta*torch.ones(subspace[0].size(-2), device=device)
    else:
        # should error
        pass

    if len(lam) > 0:
        lam = torch.tensor(lam, dtype=torch.float32, device=device)
        if len(lam) == 1:
            lam = lam*torch.ones(subspace[0].size(-2), device=device)
    else:
        # should error
        pass
    
    
    
    recovered_sources = []
    training_losses = []
    training_MISIs = []

    for seed in range(nRuns):
        # print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
        
        if data.lower() == 'mat':
            # ckpt_file = os.path.join(args.checkpoints, 'misa_{}_{}_s{}.pt'.format(data, config.dataset, seed))
            data_filename_prefix = data_filename.split('.')[0]
            ckpt_file = os.path.join(args.checkpoints, f'misa_{data}_{data_filename_prefix}_{w}_in-{os.path.split(args.filename)[1].split(".")[0]}_w-{os.path.split(args.weights)[1].split(".")[0]}_seed{seed}_lr-{lr}_bs{batch_size}_ab1{adam_betas[0]}_ab2{adam_betas[1]}.pt')
            # ckpt_file = os.path.join(args.checkpoints, "combined.pt")
        else:
            fnaming = f'misa_{subspace_name}_{data}_{config.output_filename_prefix}_in-{os.path.split(args.filename)[1].split(".")[0]}_w-{os.path.split(args.weights)[1].split(".")[0]}_seed{seed}_lr-{lr}_bs{batch_size}_ab1-{adam_betas[0]}_ab2-{adam_betas[1]}'
            ckpt_file = os.path.join(args.checkpoints, fnaming + '.pt')

        recov_sources, training_loss, training_MISI = MISA_wrapper(data_loader=train_data200,
                                                                    index=index,
                                                                    subspace=subspace, 
                                                                    eta=eta, 
                                                                    beta=beta, 
                                                                    lam=lam,
                                                                    input_dim=input_dim, 
                                                                    output_dim=output_dim, 
                                                                    seed=seed,
                                                                    epochs=epochs,
                                                                    lr=lr,
                                                                    adam_betas=adam_betas,
                                                                    weights=initial_weights,
                                                                    A=ground_truth_A,
                                                                    device=device,
                                                                    ckpt_file=ckpt_file,
                                                                    test=test,
                                                                    test_data_loader=test_data,
                                                                    train_data_loader2=train_data300)
        
        # store results
        # recovered_sources[l][n].append(recov_sources)
        recovered_sources.append(recov_sources)
        training_losses.append(training_loss)
        training_MISIs.append(training_MISI)

        # if mask_name.lower() in ['ukb2907-smri-aal2']:
        #     continue

        # results[l][n].append(np.min([metric(z, s) for z in recov_sources]))
        # print(np.min([metric(z, s) for z in recov_sources]))

    # prepare output
    if data.lower() == 'mat':
        Results = {
            # 'input_dim': input_dim,
            # 'CorrelationCoef': results,
            'recovered_sources': recovered_sources,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'loss': training_losses, 
            'MISI': training_MISIs}
    else:
        Results = {
            # 'input_dim': input_dim,
            # 'CorrelationCoef': results,
            'output_filename_prefix': config.output_filename_prefix,
            'datafile': datafile,
            'mask_name': mask_name,
            'subspace_name': subspace_name,
            'subspace': subspace,
            'recovered_sources': recovered_sources,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'adam_betas': adam_betas,
            'loss': training_losses, 
            'MISI': training_MISIs}
    # else:
    #     if mask_name.lower() in ['simtb16']:
    #         Results = {
    #             'mask_name': mask_name,
    #             'CorrelationCoef': results,
    #             'recovered_sources': recovered_sources
    #         }
    #     elif mask_name.lower() in ['ukb2907-smri-aal2']:
    #         Results = {
    #             'mask_name': mask_name,
    #             'recovered_sources': recovered_sources
    #         }
        

    return Results