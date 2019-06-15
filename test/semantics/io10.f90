! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

!OPTIONS: -Mstandard

  write(*, '(B0)')
  write(*, '(B3)')

  !ERROR: Expected 'B' edit descriptor 'w' value
  write(*, '(B)')

  !ERROR: Expected 'EN' edit descriptor 'w' value
  !ERROR: Non-standard '$' edit descriptor
  write(*, '(EN,$)')

  !ERROR: Expected 'G' edit descriptor 'w' value
  write(*, '(3G)')

  !ERROR: Non-standard '\' edit descriptor
  write(*,'(A, \)') 'Hello'

  !ERROR: 'X' edit descriptor must have a positive position value
  write(*, '(X)')
end
