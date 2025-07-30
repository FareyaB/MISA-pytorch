#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRD
#SBATCH --nodes=1
#SBATCH -c 5
#SBATCH --mem=44g
#SBATCH --array=21-23
#SBATCH -t 7200
#SBATCH -J IVA
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xli77@gsu.edu

sleep 5s

source /home/users/xli77/anaconda3/bin/activate
conda activate icebeem

cd /data/users4/xli/MISA/MISA-pytorch

if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/lr0.007_bs200_ab1-0.95_ab2-0.87 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/lr0.007_bs200_ab1-0.95_ab2-0.87 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 2 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/lr0.007_bs200_ab1-0.95_ab2-0.87 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 3 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva2.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/lr0.0265_bs316_ab1-0.95_ab2-0.81 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 4 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva2.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/lr0.0265_bs316_ab1-0.95_ab2-0.81 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 5 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva2.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/lr0.0265_bs316_ab1-0.95_ab2-0.81 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 6 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva3.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/lr0.1_bs200_ab1-0.919_ab2-0.81 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 7 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva3.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/lr0.1_bs200_ab1-0.919_ab2-0.81 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 8 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva3.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/lr0.1_bs200_ab1-0.919_ab2-0.81 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 9 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva4.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/lr0.0093_bs316_ab1-0.9233_ab2-0.87 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 10 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva4.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/lr0.0093_bs316_ab1-0.9233_ab2-0.87 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 11 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva4.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/lr0.0093_bs316_ab1-0.9233_ab2-0.87 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 12 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva5.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/lr0.029_bs316_ab1-0.95_ab2-0.8484 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 13 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva5.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/lr0.029_bs316_ab1-0.95_ab2-0.8484 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 14 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva5.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/lr0.029_bs316_ab1-0.95_ab2-0.8484 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 15 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva6.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/lr0.1_bs200_ab1-0.899_ab2-0.8154 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 16 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva6.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/lr0.1_bs200_ab1-0.899_ab2-0.8154 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 17 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva6.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/lr0.1_bs200_ab1-0.899_ab2-0.8154 -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 18 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva3.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/combined_significant_model -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 19 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva3.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/combined_significant_model -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 20 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva3.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/combined_significant_model -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 21 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva6.yaml -f sim-siva_dataset100_source12_sample32768_seed7.mat -r runs/BHI25/combined_full_model -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 22 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva6.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/combined_full_model -w wpca -a
elif [[ "$SLURM_ARRAY_TASK_ID" == 23 ]]; then
    python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva6.yaml -f sim-siva_dataset100_source12_sample32768_seed21.mat -r runs/BHI25/combined_full_model -w wpca -a
fi

sleep 5s