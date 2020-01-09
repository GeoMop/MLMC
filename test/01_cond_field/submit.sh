#!/bin/bash

set -x

py_script=`pwd`/$1
pbs_script=`pwd`/$1.pbs
script_path=${py_script%/*}

output_prefix="mlmc"

cat >$pbs_script <<EOF
#!/bin/bash 
#PBS -S /bin/bash
#PBS -l select=1:ncpus=1:cgroups=cpuacct:mem=8GB -l walltime=48:00:00
#PBS -q charon
#PBS -N MLMC_vec
#PBS -j oe

cd ${script_path}
module load python36-modules-gcc
module load hdf5-1.10.0-gcc
module use /storage/praha1/home/jan-hybs/modules
module load flow123d
python3.6 ${py_script} -r -k run /storage/liberec3-tul/home/martin_spetlik/MLMC_vec_flow/test/01_cond_field
EOF

qsub $pbs_script
