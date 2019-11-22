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


include(CheckIncludeFileCXX)

check_include_file_cxx(sys/file.h HAVE_SYS_FILE_H)
check_include_file_cxx(sys/mman.h HAVE_SYS_MMAN_H)
check_include_file_cxx(sys/stat.h HAVE_SYS_STAT_H)
check_include_file_cxx(sys/types.h HAVE_SYS_TYPES_H)
check_include_file_cxx(sys/wait.h HAVE_SYS_WAIT_H)
check_include_file_cxx(unistd.h HAVE_UNISTD_H)
