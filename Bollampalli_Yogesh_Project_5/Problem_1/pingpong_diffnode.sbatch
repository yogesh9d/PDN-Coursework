#!/bin/bash
#SBATCH --partition=oucspdn_cpu
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#
#SBATCH --time=00:15:00
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=yogesh.bollampalli-1@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --job-name=hw5prob1_diff
#
#SBATCH --chdir=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_1
#SBATCH --output=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/_P5-1_diff_%J_stdout.txt
#SBATCH --error=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/_P5-1_diff_%J_stderr.txt
#
#################################################

echo "Working directory:"
pwd
echo ""

module load intel
module load GCC
module load OpenMPI

mpirun -n 2 ./pingpong_MPI 1000000 time_1M_diff.csv
mpirun -n 2 ./pingpong_MPI 2000000 time_2M_diff.csv
mpirun -n 2 ./pingpong_MPI 4000000 time_4M_diff.csv
mpirun -n 2 ./pingpong_MPI 8000000 time_8M_diff.csv
# add for other sizes
