// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_BURNSIDE_COMMON_H_
#define FORTRAN_BURNSIDE_COMMON_H_

#include "../evaluate/expression.h"

namespace Fortran::burnside {

using Expression = evaluate::Expr<evaluate::SomeType>;
using CallArguments = std::vector<Expression>;

enum InputOutputCallType {
  InputOutputCallBackspace = 11,
  InputOutputCallClose,
  InputOutputCallEndfile,
  InputOutputCallFlush,
  InputOutputCallInquire,
  InputOutputCallOpen,
  InputOutputCallPrint,
  InputOutputCallRead,
  InputOutputCallRewind,
  InputOutputCallWait,
  InputOutputCallWrite,
  InputOutputCallSIZE = InputOutputCallWrite - InputOutputCallBackspace + 1
};

using IOCallArguments = CallArguments;

enum RuntimeCallType {
  RuntimeCallFailImage = 31,
  RuntimeCallStop,
  RuntimeCallPause,
  RuntimeCallFormTeam,
  RuntimeCallEventPost,
  RuntimeCallEventWait,
  RuntimeCallSyncAll,
  RuntimeCallSyncImages,
  RuntimeCallSyncMemory,
  RuntimeCallSyncTeam,
  RuntimeCallLock,
  RuntimeCallUnlock,
  RuntimeCallSIZE = RuntimeCallUnlock - RuntimeCallFailImage + 1
};

using RuntimeCallArguments = CallArguments;

} // Fortran::burnside

#endif  // FORTRAN_BURNSIDE_COMMON_H_
