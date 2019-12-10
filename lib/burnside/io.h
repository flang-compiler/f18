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

#ifndef FORTRAN_BURNSIDE_IO_H_
#define FORTRAN_BURNSIDE_IO_H_

namespace mlir {
class OpBuilder;
class Location;
class ValueRange;
}

namespace Fortran {

namespace parser {
struct BackspaceStmt;
struct CloseStmt;
struct EndfileStmt;
struct FlushStmt;
struct InquireStmt;
struct OpenStmt;
struct PrintStmt;
struct ReadStmt;
struct RewindStmt;
struct WriteStmt;
}

/// Experimental IO lowering to FIR + runtime. The Runtime design is under
/// design.
/// FIXME This interface is also not final. Should it be based on parser::..
/// nodes and lower expressions as needed or should it get every expression
/// already lowered as mlir::Value* ? (currently second options, not sure it
/// will provide enough information for complex IO statements).
namespace burnside {

class AbstractConverter;
class BridgeImpl;

void genBackspaceStatement(AbstractConverter &, const parser::BackspaceStmt &);
void genCloseStatement(AbstractConverter &, const parser::CloseStmt &);
void genEndfileStatement(AbstractConverter &, const parser::EndfileStmt &);
void genFlushStatement(AbstractConverter &, const parser::FlushStmt &);
void genInquireStatement(AbstractConverter &, const parser::InquireStmt &);
void genOpenStatement(AbstractConverter &, const parser::OpenStmt &);
void genPrintStatement(AbstractConverter &, const parser::PrintStmt &);
void genReadStatement(AbstractConverter &, const parser::ReadStmt &);
void genRewindStatement(AbstractConverter &, const parser::RewindStmt &);
void genWriteStatement(AbstractConverter &, const parser::WriteStmt &);

}
}

#endif  // FORTRAN_BURNSIDE_IO_H_
