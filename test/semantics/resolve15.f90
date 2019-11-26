! Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

module m
  real :: var
  interface i
    !ERROR: 'var' is not a subprogram
    procedure :: sub, var
    !ERROR: Procedure 'bad' not found
    procedure :: bad
  end interface
  interface operator(.foo.)
    !ERROR: 'var' is not a subprogram
    procedure :: sub, var
    !ERROR: Procedure 'bad' not found
    procedure :: bad
  end interface
contains
  subroutine sub
  end
end

subroutine s
  interface i
    !ERROR: 'sub' is not a module procedure
    module procedure :: sub
  end interface
  interface assignment(=)
    !ERROR: 'sub' is not a module procedure
    module procedure :: sub
  end interface
contains
  subroutine sub(x, y)
    real, intent(out) :: x
    logical, intent(in) :: y
  end
end
