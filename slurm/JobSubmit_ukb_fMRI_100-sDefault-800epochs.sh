#!/bin/bash
#SBATCH -e /data/users3/rsilva/MISA-pytorch/slurm_logs/error%A-%a.err
#SBATCH -o /data/users3/rsilva/MISA-pytorch/slurm_logs/out%A-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rsilva@gsu.edu
#SBATCH --chdir=/data/users3/rsilva/MISA-pytorch
#
#SBATCH -p qTRDHM
#SBATCH --account=trends53c17
#SBATCH --job-name=torchMISA
#SBATCH --verbose
#SBATCH --time=7200
#
#SBATCH --nodes=1
#SBATCH --mem=1280g
#SBATCH --cpus-per-task=96
#SBATCH --nodelist=arctrdhm002


# Other options:
# --gres=gpu:A100:1


sleep 5s

###########################################
##   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ##
##                                       ##
##   UPDATE CHDIR DIRECTIVE ABOVE !!!!   ## <<<<<------------------- !!!!! ****
##                                       ##
##   |||||||||||||||||||||||||||||||||   ##
###########################################

hostname
pwd

source ~/.bashrc
. ~/init_miniconda3.sh
which conda
conda activate al3118

# echo current active conda environment:
echo "Current conda environment: $CONDA_DEFAULT_ENV"
conda info

experimenter="$USER"

data_format="fMRI"

data_file="./slurm/ukb_fMRIpaths_unaffected_100.txt"

configuration="ukb-fmri-iva-100-sDefault-800epochs.yaml"

W="/data/users3/rsilva/dask-iva/W_gpca_M100_K12_ukb_fMRI.npy"

results_path="results/2025bhi-rebuttal/"

# echo parameters, including variable name and value:
echo ""
echo "slurm_job_id: $SLURM_JOB_ID"
echo "experimenter: $experimenter"
echo "data_file: $data_file"
echo "configuration: $configuration"
echo "W: $W"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "slurm_job_name: $SLURM_JOB_NAME"
echo "slurm_job_cpus_per_node: $SLURM_JOB_CPUS_PER_NODE"
echo "slurm_job_partition: $SLURM_JOB_PARTITION"
echo "slurm_job_mem: $SLURM_MEM_PER_NODE"

echo ""
echo "Running MISA with the following command:"
echo "python -u main.py --data $data_format --config $configuration --filename $data_file --run $results_path --weights $W"

# Run the main script with the specified configuration, data file, results path, and W matrix:
python -u main.py --data "$data_format" --config "$configuration" --filename "$data_file" --run "$results_path" --weights "$W"

# python3 main.py -c /data/users4/xli/MISA/MISA-pytorch/configs/sim-iva.yaml -f sim-siva_dataset100_source12_sample32768_seed14.mat -r runs/BHI25/lr0.007_bs200_ab1-0.95_ab2-0.87 -w

echo ""
echo "FINISHED... preparing to exit."

# echo parameters, including variable name and value:
echo ""
echo "slurm_job_id: $SLURM_JOB_ID"
echo "experimenter: $experimenter"
echo "data_file: $data_file"
echo "configuration: $configuration"
echo "W: $W"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "slurm_job_name: $SLURM_JOB_NAME"
echo "slurm_job_cpus_per_node: $SLURM_JOB_CPUS_PER_NODE"
echo "slurm_job_partition: $SLURM_JOB_PARTITION"
echo "slurm_job_mem: $SLURM_MEM_PER_NODE"

sleep 5s