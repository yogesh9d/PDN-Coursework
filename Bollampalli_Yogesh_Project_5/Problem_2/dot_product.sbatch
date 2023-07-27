#!/bin/bash
#SBATCH --partition=oucspdn_cpu
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=8
#
#SBATCH --time=00:15:00
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=yogesh.bollampalli-1@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --job-name=hw5prob2
#
#SBATCH --chdir=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_2
#SBATCH --output=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_2/_P5-2_%J_stdout.txt
#SBATCH --error=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_2/_P5-2_%J_stderr.txt
#
#################################################

echo "Working directory:"
pwd
echo ""

module load intel
module load GCC
module load OpenMPI

mpirun -n 2 ./dot_product_MPI 262144 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^18.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^18.csv result_2p_18.csv time_2p_18.csv
mpirun -n 2 ./dot_product_MPI 524288 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^19.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^19.csv result_2p_19.csv time_2p_19.csv
mpirun -n 2 ./dot_product_MPI 1048576 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^20.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^20.csv result_2p_20.csv time_2p_20.csv

mpirun -n 4 ./dot_product_MPI 262144 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^18.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^18.csv result_4p_18.csv time_4p_18.csv
mpirun -n 4 ./dot_product_MPI 524288 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^19.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^19.csv result_4p_19.csv time_4p_19.csv
mpirun -n 4 ./dot_product_MPI 1048576 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^20.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^20.csv result_4p_20.csv time_4p_20.csv

mpirun -n 8 ./dot_product_MPI 262144 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^18.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^18.csv result_8p_18.csv time_8p_18.csv
mpirun -n 8 ./dot_product_MPI 524288 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^19.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^19.csv result_8p_19.csv time_8p_19.csv
mpirun -n 8 ./dot_product_MPI 1048576 /home/oucspdn030/Project_5/test_data/Vectors/vec1_2^20.csv /home/oucspdn030/Project_5/test_data/Vectors/vec2_2^20.csv result_8p_20.csv time_8p_20.csv


# add for other sizes
