#!/bin/bash

set -x

py_script=`pwd`/$1
pbs_script=`pwd`/$1.pbs
script_path=${py_script%/*}

work_dir=$2
mlmc=/storage/liberec3-tul/home/martin_spetlik/MLMC_new_design

cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=1:cgroups=cpuacct:mem=128mb
#PBS -q charon_2h
#PBS -N MLMC
#PBS -j oe

cd ${work_dir}

echo "Load modules..."

module use /storage/praha1/home/jan-hybs/modules
module load flow123d

python3 -m venv env
source env/bin/activate
pip3 install pyyaml attrs numpy scipy h5py ${mlmc}

cd ${script_path}

#pip3 freeze
#module list
python3 ${py_script} ${work_dir}
EOF

#qsub $pbs_script
sh .$pbs_script
