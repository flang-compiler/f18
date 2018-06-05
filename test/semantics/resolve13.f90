! Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

module m1
  integer :: x
  integer, private :: y
end

use m1, local_x => x
!ERROR: 'y' is PRIVATE in 'm1'
use m1, local_y => y
!ERROR: 'z' not found in module 'm1'
use m1, local_z => z

!ERROR: 'y' is PRIVATE in 'm1'
use m1, only: y
!ERROR: 'z' not found in module 'm1'
use m1, only: z
!ERROR: 'z' not found in module 'm1'
use m1, only: my_x => z

end
