#!/bin/bash
#SBATCH -e /data/users3/rsilva/MISA-pytorch/slurm_logs/error%A-%a.err
#SBATCH -o /data/users3/rsilva/MISA-pytorch/slurm_logs/out%A-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rsilva@gsu.edu
#SBATCH --chdir=/data/users3/rsilva/MISA-pytorch
#
#SBATCH -p qTRDHM
#SBATCH --account=trends53c17
#SBATCH --job-name=IVAGL
#SBATCH --verbose
#SBATCH --time=7200
#
#SBATCH --nodes=1
#SBATCH --mem=1280g
#SBATCH --cpus-per-task=96
#SBATCH --nodelist=arctrdhm003


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
module load matlab/R2025a

# echo current active conda environment:
# echo "Current conda environment: $CONDA_DEFAULT_ENV"
# conda info

experimenter="$USER"

# echo parameters, including variable name and value:
echo ""
echo "slurm_job_id: $SLURM_JOB_ID"
echo "experimenter: $experimenter"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "slurm_job_name: $SLURM_JOB_NAME"
echo "slurm_job_cpus_per_node: $SLURM_JOB_CPUS_PER_NODE"
echo "slurm_job_partition: $SLURM_JOB_PARTITION"
echo "slurm_job_mem: $SLURM_MEM_PER_NODE"


echo "Running GIFT IVA-GL script in MATLAB with the following command:"
echo "matlab -nodisplay -nojvm -batch \"GIFT_IVAGL_run; exit\""

# Run GIFT IVA-GL script in MATLAB:
# ******** NOTE: USE THE APPROPRIATE CHDIR DIRECTIVE ABOVE, so your .m script is visible to MATLAB!!!
matlab -nodisplay -nojvm -batch "GIFT_IVAGL_run; exit"
# Alternatively:
# matlab -nodisplay -nojvm -batch "addpath('/data/users3/rsilva/MISA-pytorch'); GIFT_IVAGL_run; exit"

echo ""
echo "FINISHED... preparing to exit."


sleep 5s