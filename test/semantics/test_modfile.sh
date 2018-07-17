#!/usr/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Compile a source file and compare generated .mod files against expected.

set -e
PATH=/usr/bin
srcdir=$(dirname $0)
CMD="${F18:-../../../tools/f18/f18} -fdebug-resolve-names -fparse-only"

if [[ $# != 1 ]]; then
  echo "Usage: $0 <fortran-source>"
  exit 1
fi
src=$srcdir/$1
[[ ! -f $src ]] && echo "File not found: $src" && exit 1

if [[ $KEEP ]]; then
  temp=.
else
  temp=$(mktemp --directory --tmpdir=.)
  trap "rm -rf $temp" EXIT
fi

( cd $temp && $CMD $src )

actual=$temp/actual.mod
expect=$temp/expect.mod
actual_files=$temp/actual_files
diffs=$temp/diffs

( cd $temp && ls -1 *.mod ) > $actual_files
expected_files=$(sed -n 's/^!Expect: \(.*\)/\1/p' $src)
extra_files=$(echo "$expected_files" | comm -23 $actual_files -)
if [[ ! -z "$extra_files" ]]; then
  echo "Unexpected .mod files produced:" $extra_files
  echo FAIL
  exit 1
fi
for mod in $expected_files; do
  if [[ ! -f $temp/$mod ]]; then
    echo "Compilation did not produce expected mod file: $mod"
    echo FAIL
    exit 1
  fi
  sed '/^!mod\$/d' $temp/$mod > $actual
  sed '1,/^!Expect: '"$mod"'/d' $src | sed -e '/^$/,$d' -e 's/^!//' > $expect
  if ! diff -U999999 $actual $expect > $diffs; then
    echo "Module file $mod differs from expected:"
    sed '1,2d' $diffs
    echo FAIL
    exit 1
  fi
done
echo PASS
