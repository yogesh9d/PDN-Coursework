#!/bin/bash
#SBATCH --partition=oucspdn_cpu
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#
#SBATCH --time=00:15:00
#SBATCH --job-name=hw5prob3
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=yogesh.bollampalli-1@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --chdir=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_3
#SBATCH --output=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_3/_P5-3_%J_stdout.txt
#SBATCH --error=/home/oucspdn030/Project_5/Yogesh_Bollampalli_Project_5/Problem_3/_P5-3_%J_stderr.txt

#################################################

echo "Working directory:"
pwd
echo ""

module load intel
module load OpenMPI

mpirun -n 2 ./merge_sort_MPI 262144 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^18.csv result_1_diff_18.csv time_1_diff_18.csv
mpirun -n 4 ./merge_sort_MPI 262144 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^18.csv result_2_diff_18.csv time_2_diff_18.csv
mpirun -n 8 ./merge_sort_MPI 262144 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^18.csv result_4_diff_18.csv time_4_diff_18.csv

mpirun -n 2 ./merge_sort_MPI 524288 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^19.csv result_1_diff_19.csv time_1_diff_19.csv
mpirun -n 4 ./merge_sort_MPI 524288 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^19.csv result_2_diff_19.csv time_2_diff_19.csv
mpirun -n 8 ./merge_sort_MPI 524288 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^19.csv result_4_diff_19.csv time_4_diff_19.csv

mpirun -n 2 ./merge_sort_MPI 1048576 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^20.csv result_1_diff_20.csv time_1_diff_20.csv
mpirun -n 4 ./merge_sort_MPI 1048576 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^20.csv result_2_diff_20.csv time_2_diff_20.csv
mpirun -n 8 ./merge_sort_MPI 1048576 /home/oucspdn030/Project_5/test_data/Problem_3/sorted_vec1_2^20.csv result_4_diff_20.csv time_4_diff_20.csv

# add more here ...
