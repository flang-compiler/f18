//===-- lib/lower/io.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_IO_H_
#define FORTRAN_LOWER_IO_H_

namespace mlir {
class OpBuilder;
class Location;
class ValueRange;
} // namespace mlir

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
} // namespace parser

/// Experimental IO lowering to FIR + runtime. The Runtime design is under
/// design.
/// FIXME This interface is also not final. Should it be based on parser::..
/// nodes and lower expressions as needed or should it get every expression
/// already lowered as mlir::Value? (currently second options, not sure it
/// will provide enough information for complex IO statements).
namespace lower {

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

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_IO_H_
