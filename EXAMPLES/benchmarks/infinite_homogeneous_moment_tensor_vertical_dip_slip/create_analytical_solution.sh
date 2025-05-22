#!/bin/bash
#
#

currentdir=`pwd`

# gets the default compiler
f90=`grep "^FC " ../../../Makefile | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`

# selects compiler
#f90=gfortran
#f90=ifort
flags="-O2"  # debug: -Wall -g -fbounds-check

# solution output
mkdir -p REF_ANALYTICAL/


# components 1 == X / 3 == Z
components=( "1" "3" )

for comp in ${components[@]}; do

echo
echo "component: $comp"

# sets component 1 == X / 3 == Z
sed -i "s:^displacement_direction_index = .*:displacement_direction_index = $comp:" compute_line_source_solution_from_point_source_solution_for_moment_tensor_in_3D.f90

echo "  compiling ..."
$f90 $flags -o xcompute_solution  compute_line_source_solution_from_point_source_solution_for_moment_tensor_in_3D.f90 

# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

echo "  running analytical solution..."
./xcompute_solution

# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi


# stores output
mv -v source_time_function.dat REF_ANALYTICAL/
if [ "$comp" == "1" ]; then
  mv -v x_displacement.dat REF_ANALYTICAL/x_displacement.dat
elif [ "$comp" == "2" ]; then
  mv -v x_displacement.dat REF_ANALYTICAL/y_displacement.dat 
elif [ "$comp" == "3" ]; then
  mv -v x_displacement.dat REF_ANALYTICAL/z_displacement.dat
fi

echo

done


echo
echo "see results in directory: REF_ANALYTICAL/"
echo
echo "done"
echo

