import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.MISAK import MISA

def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, seed, epochs, lr, adam_betas,
                 weights=list(), A=None, device='cpu', ckpt_file='misa.pt', test=False, test_data_loader=None, train_data_loader2=None):
    
    model=MISA(weights=weights,
                 index=index, 
                 subspace=subspace, 
                 eta=eta, 
                 beta=beta, 
                 lam=lam, 
                 input_dim=input_dim, 
                 output_dim=output_dim,
                 seed=seed,
                 device=device)
    
    model.to(device=device)
    
    final_MISI = []
    
    if not test:
        test_loss = model.predict(test_data_loader)
        print(f"\nINITIAL test loss: {test_loss[0].numpy():.3f}\n")

        print("Initial weights:")
        print(f"mm = 0: {model.net[0].weight[:3,:4]}")
        print(f"mm = 1: {model.net[1].weight[:3,:4]}")
        print()

        training_loss, training_MISI, optimizer, final_weights = model.train_me(data_loader, epochs, lr, adam_betas, A, train_data_loader2)
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        
        test_loss = model.predict(test_data_loader)
        print(f"\nFINAL test loss: {test_loss[0].numpy():.3f}\n")

        print("Final weights:")
        print(f"mm = 0: {model.net[0].weight[:3,:4]}")
        print(f"mm = 1: {model.net[1].weight[:3,:4]}")
        print()

        torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'seed': model.seed,
                'index': model.index,
                'subspace': model.subspace,
                'eta': model.eta,
                'beta': model.beta,
                'lam': model.lam,
                'training_loss': training_loss,
                'training_MISI': training_MISI,
                'test_loss': test_loss},
               ckpt_file)
        print("Saved checkpoint to: " + ckpt_file)

        fnaming = os.path.split(ckpt_file)[1].split('.pt')[0]
        np.save(os.path.join(os.path.dirname(ckpt_file), f'{fnaming}_final-weights.npy'), final_weights)
        print("Saved final weights to: " + os.path.join(os.path.dirname(ckpt_file), f'{fnaming}_final-weights.npy'))

    else:
        checkpoint = torch.load(ckpt_file)
        model.load_state_dict(checkpoint['model'])
        test_loss = model.predict(test_data_loader)
        print(f"test loss: {test_loss[0].numpy():.3f}")
    
    return model.output, training_loss, training_MISI