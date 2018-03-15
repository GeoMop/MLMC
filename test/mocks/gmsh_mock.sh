#!/bin/bash

set -x 

script_dir=${0%/*}

shift #-2
shift #-clscale
echo "Mesh step: $1"
shift

shift #-o
mesh_file=$1
shift
geo_file=$1
cp ${script_dir}/mock_mesh.msh ${mesh_file}
 