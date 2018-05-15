#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -q charon
#PBS -N Flow123d
#PBS -j oe
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b955-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b955-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b955-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b956-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b956-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b956-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b957-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b957-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b957-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b958-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b958-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b958-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b959-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b959-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b959-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b95a-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b95a-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b95a-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b95b-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b95b-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b95b-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b95c-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b95c-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b95c-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b95d-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b95d-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b95d-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b95e-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b95e-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b95e-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b95f-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b95f-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b95f-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b960-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b960-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b960-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b961-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b961-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b961-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b962-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b962-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b962-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b963-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b963-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b963-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b964-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b964-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b964-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b965-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b965-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b965-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b966-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b966-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b966-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b967-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_2_step_0.282843/flow_input.yaml  -o samples/98c4b967-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b967-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b968-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b968-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b968-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b969-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b969-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b969-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b96a-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b96a-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b96a-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b96b-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b96b-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b96b-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b96c-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b96c-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b96c-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b96d-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b96d-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b96d-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b96e-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b96e-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b96e-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b96f-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b96f-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b96f-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b970-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b970-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b970-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b971-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b971-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b971-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b972-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b972-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b972-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b973-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b973-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b973-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b974-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b974-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b974-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b975-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b975-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b975-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b976-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b976-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b976-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b977-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b977-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b977-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b978-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b978-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b978-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b979-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b979-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b979-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b97a-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_4_step_0.080000/flow_input.yaml  -o samples/98c4b97a-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b97a-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
cd /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424
time -p /storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d --yaml_balance -i samples/98c4b97b-5825-11e8-aea4-0050568e4d1d -s /auto/liberec1-tul/martin_spetlik/MLMC_flow/test/01_cond_field/sim_3_step_0.150424/flow_input.yaml  -o samples/98c4b97b-5825-11e8-aea4-0050568e4d1d >/dev/null 2>&1
cd samples/98c4b97b-5825-11e8-aea4-0050568e4d1d
touch FINISHED
rm profiler*
rm flow123.0.log
rm flow_fields.msh
rm fields_sample.msh
rm water_balance.txt
