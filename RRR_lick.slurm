#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30gb
#SBATCH --cpus-per-task=1
#SBATCH --array=1-51
#SBATCH --time=4-00:00:00

# Load necessary modules
cd
source software/init-conda
conda activate allenSDK
cd codes

input=$(sed -n "${SLURM_ARRAY_TASK_ID}p" familiar_inputs.txt)
# Call your Python script with the input parameter
python RRR_lick.py --input $input

