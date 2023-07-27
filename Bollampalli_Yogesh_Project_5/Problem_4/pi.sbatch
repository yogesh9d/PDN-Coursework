#!/bin/bash
#SBATCH --partition=oucspdn_cpu
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#
#SBATCH --time=00:15:00
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=yogesh.bollampalli-1@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --job-name=hw5prob4
#
#SBATCH --chdir=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_4
#SBATCH --output=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_4/_P5-4_%J_stdout.txt
#SBATCH --error=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_4/_P5-4_%J_stderr.txt
#
#################################################

echo "Working directory:"
pwd
echo ""

module load intel
module load OpenMPI

mpirun -n 1 ./pi_MPI result_1p.csv time_1p.
mpirun -n 2 ./pi_MPI result_2p.csv time_2p.
mpirun -n 4 ./pi_MPI result_4p.csv time_4p.
mpirun -n 8 ./pi_MPI result_8p.csv time_8p.
mpirun -n 16 ./pi_MPI result_16p.csv time_16p.

# Add more here
