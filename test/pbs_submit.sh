#!/bin/bash

set -x

py_script=`pwd`/$1
pbs_script=`pwd`/$1.pbs
script_path=${py_script%/*}

work_dir=$2
mlmc=/storage/liberec3-tul/home/martin_spetlik/MLMC_quantity

cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=1:cgroups=cpuacct:mem=16Gb:scratch_local=2gb
#PBS -q charon
#PBS -N MLMC
#PBS -j oe

#export TMPDIR=$SCRATCHDIR

cd ${work_dir}
module load python36-modules-gcc

python3 -m venv env --clear
source env/bin/activate

python3 -m pip install attrs numpy scipy h5py gstools ruamel.yaml memoization ${mlmc}

cd ${script_path}

python3 ${py_script} run ${work_dir} --clean
deactivate
EOF

qsub $pbs_script
