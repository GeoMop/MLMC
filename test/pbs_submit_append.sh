#!/bin/bash

set -x

py_script=`pwd`/$1
pbs_script=`pwd`/$1.pbs
script_path=${py_script%/*}

work_dir=$2
mlmc=/auto/liberec3-tul/home/martin_spetlik/MLMC_fix_bugs/MLMC

cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=1:cgroups=cpuacct:mem=6Gb
#PBS -l walltime=48:00:00
#PBS -q charon
#PBS -N MLMC
#PBS -j oe

#export TMPDIR=$SCRATCHDIR

cd ${work_dir}
module load python/3.8.0-gcc

python3.8 -m venv env --clear
source env/bin/activate

python3.8 -m pip install attrs numpy scipy h5py gstools ruamel.yaml sklearn memoization seaborn ${mlmc}

cd ${script_path}

python3.8 ${py_script} run ${work_dir} --debug
deactivate
EOF

qsub $pbs_script
