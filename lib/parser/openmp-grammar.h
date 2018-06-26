// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_PARSER_OPENMP_GRAMMAR_H_
#define FORTRAN_PARSER_OPENMP_GRAMMAR_H_

// Top-level grammar specification for OpenMP.
// See OpenMP-4.5-grammar.txt for documentation.

#include "basic-parsers.h"
#include "characters.h"
#include "debug-parser.h"
#include "grammar.h"
#include "parse-tree.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parsers.h"
#include "user-state.h"
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

// OpenMP Directives and Clauses
namespace Fortran::parser {

constexpr auto endOmpLine = space >> endOfLine;

// OpenMP Clauses
// DEFAULT (PRIVATE | FIRSTPRIVATE | SHARED | NONE )
TYPE_PARSER(construct<OmpDefaultClause>(
    "PRIVATE" >> pure(OmpDefaultClause::Type::Private) ||
    "FIRSTPRIVATE" >> pure(OmpDefaultClause::Type::Firstprivate) ||
    "SHARED" >> pure(OmpDefaultClause::Type::Shared) ||
    "NONE" >> pure(OmpDefaultClause::Type::None)))

// PROC_BIND(CLOSE | MASTER | SPREAD)
TYPE_PARSER(construct<OmpProcBindClause>(
    "CLOSE" >> pure(OmpProcBindClause::Type::Close) ||
    "MASTER" >> pure(OmpProcBindClause::Type::Master) ||
    "SPREAD" >> pure(OmpProcBindClause::Type::Spread)))

// MAP ([ [map-type-modifier[,]] map-type : ] list)
// map-type-modifier -> ALWAYS
// map-type -> TO | FROM | TOFROM | ALLOC | RELEASE | DELETE
TYPE_PARSER(construct<OmpMapType>(
    maybe("ALWAYS"_tok >> construct<OmpMapType::Always>() / maybe(","_tok)),
    "TO"_tok >> pure(OmpMapType::Type::To) / ":"_tok ||
        "FROM"_tok >> pure(OmpMapType::Type::From) / ":"_tok ||
        "TOFROM"_tok >> pure(OmpMapType::Type::Tofrom) / ":"_tok ||
        "ALLOC"_tok >> pure(OmpMapType::Type::Alloc) / ":"_tok ||
        "RELEASE"_tok >> pure(OmpMapType::Type::Release) / ":"_tok ||
        "DELETE"_tok >> pure(OmpMapType::Type::Delete) / ":"_tok))

TYPE_PARSER(construct<OmpMapClause>(
    maybe(Parser<OmpMapType>{}), Parser<OmpObjectList>{}))

// SCHEDULE ([modifier [, modifier]:]kind[, chunk_size])
// Modifier ->  MONITONIC | NONMONOTONIC | SIMD
// kind -> STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
// chunk_size -> ScalarIntExpr
TYPE_PARSER(construct<OmpScheduleModifierType>(
    "MONOTONIC" >> pure(OmpScheduleModifierType::ModType::Monotonic) ||
    "NONMONOTONIC" >> pure(OmpScheduleModifierType::ModType::Nonmonotonic) ||
    "SIMD" >> pure(OmpScheduleModifierType::ModType::Simd)))

TYPE_PARSER(construct<OmpScheduleModifier>(Parser<OmpScheduleModifierType>{},
    maybe(","_tok >> Parser<OmpScheduleModifierType>{})))

TYPE_PARSER(construct<OmpScheduleClause>(maybe(Parser<OmpScheduleModifier>{}),
    "STATIC" >> pure(OmpScheduleClause::ScheduleType::Static) ||
        "DYNAMIC" >> pure(OmpScheduleClause::ScheduleType::Dynamic) ||
        "GUIDED" >> pure(OmpScheduleClause::ScheduleType::Guided) ||
        "AUTO" >> pure(OmpScheduleClause::ScheduleType::Auto) ||
        "RUNTIME" >> pure(OmpScheduleClause::ScheduleType::Runtime),
    maybe(","_tok >> scalarIntExpr)))

// IF(directive-name-modifier: scalar-logical-expr)
TYPE_PARSER(construct<OmpIfClause>(
    maybe(
        ("PARALLEL"_tok >> pure(OmpIfClause::DirectiveNameModifier::Parallel) ||
            "TARGET ENTER DATA"_tok >>
                pure(OmpIfClause::DirectiveNameModifier::TargetEnterData) ||
            "TARGET EXIT DATA"_tok >>
                pure(OmpIfClause::DirectiveNameModifier::TargetExitData) ||
            "TARGET DATA"_tok >>
                pure(OmpIfClause::DirectiveNameModifier::TargetData) ||
            "TARGET UPDATE"_tok >>
                pure(OmpIfClause::DirectiveNameModifier::TargetUpdate) ||
            "TARGET"_tok >> pure(OmpIfClause::DirectiveNameModifier::Target) ||
            "TASKLOOP"_tok >>
                pure(OmpIfClause::DirectiveNameModifier::Taskloop) ||
            "TASK"_tok >> pure(OmpIfClause::DirectiveNameModifier::Task)) /
        ":"_tok),
    scalarLogicalExpr))

// REDUCTION(reduction-identifier: list)
constexpr auto reductionBinaryOperator = "+" >>
        pure(OmpReductionOperator::BinaryOperator::Add) ||
    "-" >> pure(OmpReductionOperator::BinaryOperator::Subtract) ||
    "*" >> pure(OmpReductionOperator::BinaryOperator::Multiply) ||
    ".AND." >> pure(OmpReductionOperator::BinaryOperator::AND) ||
    ".OR." >> pure(OmpReductionOperator::BinaryOperator::OR) ||
    ".EQV." >> pure(OmpReductionOperator::BinaryOperator::EQV) ||
    ".NEQV." >> pure(OmpReductionOperator::BinaryOperator::NEQV);

constexpr auto reductionProcedureOperator = "MIN" >>
        pure(OmpReductionOperator::ProcedureOperator::MIN) ||
    "MAX" >> pure(OmpReductionOperator::ProcedureOperator::MAX) ||
    "IAND" >> pure(OmpReductionOperator::ProcedureOperator::IAND) ||
    "IOR" >> pure(OmpReductionOperator::ProcedureOperator::IOR) ||
    "IEOR" >> pure(OmpReductionOperator::ProcedureOperator::IEOR);

TYPE_PARSER(construct<OmpReductionOperator>(reductionBinaryOperator) ||
    construct<OmpReductionOperator>(reductionProcedureOperator))

TYPE_PARSER(construct<OmpReductionClause>(
    Parser<OmpReductionOperator>{} / ":"_tok, nonemptyList(designator)))

// DEPEND(SOURCE | SINK : vec | (IN | OUT | INOUT) : list
TYPE_PARSER(construct<OmpDependSinkVecLength>(
    indirect(Parser<DefinedOperator>{}), scalarIntConstantExpr))

TYPE_PARSER(
    construct<OmpDependSinkVec>(name, maybe(Parser<OmpDependSinkVecLength>{})))

TYPE_PARSER(construct<OmpDependenceType>(
    "INOUT"_tok >> pure(OmpDependenceType::Type::Inout) ||
    "IN"_tok >> pure(OmpDependenceType::Type::In) ||
    "OUT"_tok >> pure(OmpDependenceType::Type::Out)))

TYPE_CONTEXT_PARSER("Omp Depend clause"_en_US,
    construct<OmpDependClause>(construct<OmpDependClause::Sink>(
        "SINK"_tok >> ":"_tok >> nonemptyList(Parser<OmpDependSinkVec>{}))) ||
        construct<OmpDependClause>(
            construct<OmpDependClause::Source>("SOURCE"_tok)) ||
        construct<OmpDependClause>(construct<OmpDependClause::InOut>(
            Parser<OmpDependenceType>{}, ":"_tok >> nonemptyList(designator))))

// linear-modifier
TYPE_PARSER(construct<OmpLinearModifier>(
    "REF"_tok >> pure(OmpLinearModifier::Type::Ref) ||
    "VAL"_tok >> pure(OmpLinearModifier::Type::Val) ||
    "UVAL"_tok >> pure(OmpLinearModifier::Type::Uval)))

// LINEAR(list: linear-step)
TYPE_CONTEXT_PARSER("Omp LINEAR clause"_en_US,
    construct<OmpLinearClause>(
        construct<OmpLinearClause>(construct<OmpLinearClause::WithModifier>(
            Parser<OmpLinearModifier>{}, parenthesized(nonemptyList(name)),
            maybe(":"_tok >> scalarIntConstantExpr))) ||
        construct<OmpLinearClause>(construct<OmpLinearClause::WithoutModifier>(
            nonemptyList(name), maybe(":"_tok >> scalarIntConstantExpr)))))

// ALIGNED(list: alignment)
TYPE_PARSER(construct<OmpAlignedClause>(
    nonemptyList(name), maybe(":"_tok) >> scalarIntConstantExpr))

TYPE_PARSER(construct<OmpObject>(pure(OmpObject::Kind::Object), designator) ||
    construct<OmpObject>(
        "/" >> pure(OmpObject::Kind::Common), designator / "/"))

TYPE_PARSER("DEFAULTMAP"_tok >>
        construct<OmpClause>(construct<OmpClause::Defaultmap>(
            parenthesized("TOFROM"_tok >> ":"_tok >> "SCALAR"_tok))) ||
    "INBRANCH"_tok >> construct<OmpClause>(construct<OmpClause::Inbranch>()) ||
    "MERGEABLE"_tok >>
        construct<OmpClause>(construct<OmpClause::Mergeable>()) ||
    "NOGROUP"_tok >> construct<OmpClause>(construct<OmpClause::Nogroup>()) ||
    "NOTINBRANCH"_tok >>
        construct<OmpClause>(construct<OmpClause::Notinbranch>()) ||
    "NOWAIT"_tok >> construct<OmpClause>(construct<OmpNowait>()) ||
    "UNTIED"_tok >> construct<OmpClause>(construct<OmpClause::Untied>()) ||
    "COLLAPSE"_tok >> construct<OmpClause>(construct<OmpClause::Collapse>(
                          parenthesized(scalarIntConstantExpr))) ||
    "COPYIN"_tok >> construct<OmpClause>(construct<OmpClause::Copyin>(
                        parenthesized(Parser<OmpObjectList>{}))) ||
    "COPYPRIVATE"_tok >> construct<OmpClause>(construct<OmpClause::Copyprivate>(
                             (parenthesized(Parser<OmpObjectList>{})))) ||
    "DEVICE"_tok >> construct<OmpClause>(construct<OmpClause::Device>(
                        parenthesized(scalarIntExpr))) ||
    "DIST_SCHEDULE"_tok >>
        construct<OmpClause>(construct<OmpClause::DistSchedule>(
            parenthesized("STATIC"_tok >> ","_tok >> scalarIntExpr))) ||
    "FINAL"_tok >> construct<OmpClause>(construct<OmpClause::Final>(
                       parenthesized(scalarIntExpr))) ||
    "FIRSTPRIVATE"_tok >>
        construct<OmpClause>(construct<OmpClause::Firstprivate>(
            parenthesized(Parser<OmpObjectList>{}))) ||
    "FROM"_tok >> construct<OmpClause>(construct<OmpClause::From>(
                      parenthesized(nonemptyList(designator)))) ||
    "GRAINSIZE"_tok >> construct<OmpClause>(construct<OmpClause::Grainsize>(
                           parenthesized(scalarIntExpr))) ||
    "LASTPRIVATE"_tok >> construct<OmpClause>(construct<OmpClause::Lastprivate>(
                             parenthesized(Parser<OmpObjectList>{}))) ||
    "NUM_TASKS"_tok >> construct<OmpClause>(construct<OmpClause::NumTasks>(
                           parenthesized(scalarIntExpr))) ||
    "NUM_TEAMS"_tok >> construct<OmpClause>(construct<OmpClause::NumTeams>(
                           parenthesized(scalarIntExpr))) ||
    "NUM_THREADS"_tok >> construct<OmpClause>(construct<OmpClause::NumThreads>(
                             parenthesized(scalarIntExpr))) ||
    "ORDERED"_tok >> construct<OmpClause>(construct<OmpClause::Ordered>(
                         maybe(parenthesized(scalarIntConstantExpr)))) ||
    "PRIORITY"_tok >> construct<OmpClause>(construct<OmpClause::Priority>(
                          parenthesized(scalarIntExpr))) ||
    "PRIVATE"_tok >> construct<OmpClause>(construct<OmpClause::Private>(
                         parenthesized(Parser<OmpObjectList>{}))) ||
    "SAFELEN"_tok >> construct<OmpClause>(construct<OmpClause::Safelen>(
                         parenthesized(scalarIntConstantExpr))) ||
    "SHARED"_tok >> construct<OmpClause>(construct<OmpClause::Shared>(
                        parenthesized(Parser<OmpObjectList>{}))) ||
    "SIMDLEN"_tok >> construct<OmpClause>(construct<OmpClause::Simdlen>(
                         parenthesized(scalarIntConstantExpr))) ||
    "THREAD_LIMIT"_tok >>
        construct<OmpClause>(
            construct<OmpClause::ThreadLimit>(parenthesized(scalarIntExpr))) ||
    "TO"_tok >> construct<OmpClause>(construct<OmpClause::To>(
                    parenthesized(nonemptyList(designator)))) ||
    "UNIFORM"_tok >> construct<OmpClause>(construct<OmpClause::Uniform>(
                         parenthesized(nonemptyList(name)))) ||
    "USE_DEVICE_PTR"_tok >>
        construct<OmpClause>(construct<OmpClause::UseDevicePtr>(
            parenthesized(nonemptyList(name)))) ||
    "ALIGNED"_tok >>
        construct<OmpClause>(parenthesized(Parser<OmpAlignedClause>{})) ||
    "DEFAULT"_tok >>
        construct<OmpClause>(parenthesized(Parser<OmpDefaultClause>{})) ||
    "DEPEND"_tok >>
        construct<OmpClause>(parenthesized(Parser<OmpDependClause>{})) ||
    "IF"_tok >> construct<OmpClause>(parenthesized(Parser<OmpIfClause>{})) ||
    "LINEAR"_tok >>
        construct<OmpClause>(parenthesized(Parser<OmpLinearClause>{})) ||
    "MAP"_tok >> construct<OmpClause>(parenthesized(Parser<OmpMapClause>{})) ||
    "PROC_BIND"_tok >>
        construct<OmpClause>(parenthesized(Parser<OmpProcBindClause>{})) ||
    "REDUCTION"_tok >>
        construct<OmpClause>(parenthesized(Parser<OmpReductionClause>{})) ||
    "SCHEDULE"_tok >>
        construct<OmpClause>(parenthesized(Parser<OmpScheduleClause>{})))

// [Clause, [Clause], ...]
TYPE_PARSER(
    construct<OmpClauseList>(many(maybe(","_tok) >> Parser<OmpClause>{})))

// (variable | /common-block | array-sections)
TYPE_PARSER(construct<OmpObjectList>(nonemptyList(Parser<OmpObject>{})))

// Omp directives enclosing do loop
TYPE_PARSER("DISTRIBUTE PARALLEL DO SIMD"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::DistributeParallelDoSimd>()) ||
    "DISTRIBUTE PARALLEL DO"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::DistributeParallelDo>()) ||
    "DISTRIBUTE SIMD"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::DistributeSimd>()) ||
    "DISTRIBUTE"_tok >> construct<OmpLoopDirective>(
                            construct<OmpLoopDirective::Distribute>()) ||
    "PARALLEL DO SIMD"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::ParallelDoSimd>()) ||
    "PARALLEL DO"_tok >> construct<OmpLoopDirective>(
                             construct<OmpLoopDirective::ParallelDo>()) ||
    "SIMD"_tok >>
        construct<OmpLoopDirective>(construct<OmpLoopDirective::Simd>()) ||
    "TARGET PARALLEL DO SIMD"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TargetParallelDoSimd>()) ||
    "TARGET PARALLEL DO"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TargetParallelDo>()) ||
    "TARGET SIMD"_tok >> construct<OmpLoopDirective>(
                             construct<OmpLoopDirective::TargetSimd>()) ||
    "TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD"_tok >>
        construct<OmpLoopDirective>(construct<
            OmpLoopDirective::TargetTeamsDistributeParallelDoSimd>()) ||
    "TARGET TEAMS DISTRIBUTE PARALLEL DO"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TargetTeamsDistributeParallelDo>()) ||
    "TARGET TEAMS DISTRIBUTE SIMD"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TargetTeamsDistributeSimd>()) ||
    "TARGET TEAMS DISTRIBUTE"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TargetTeamsDistribute>()) ||
    "TASKLOOP SIMD"_tok >> construct<OmpLoopDirective>(
                               construct<OmpLoopDirective::TaskloopSimd>()) ||
    "TASKLOOP"_tok >>
        construct<OmpLoopDirective>(construct<OmpLoopDirective::Taskloop>()) ||
    "TEAMS DISTRIBUTE PARALLEL DO SIMD"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TeamsDistributeParallelDoSimd>()) ||
    "TEAMS DISTRIBUTE PARALLEL DO"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TeamsDistributeParallelDo>()) ||
    "TEAMS DISTRIBUTE SIMD"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TeamsDistributeSimd>()) ||
    "TEAMS DISTRIBUTE"_tok >>
        construct<OmpLoopDirective>(
            construct<OmpLoopDirective::TeamsDistribute>()))

// Cancellation Point construct
TYPE_PARSER("CANCELLATION POINT"_tok >>
    construct<OpenMPCancellationPointConstruct>(
        "PARALLEL"_tok >> pure(OmpCancelType::Type::Parallel) ||
        "SECTIONS"_tok >> pure(OmpCancelType::Type::Sections) ||
        "DO"_tok >> pure(OmpCancelType::Type::Do) ||
        "TASKGROUP"_tok >> pure(OmpCancelType::Type::Taskgroup)))

// Cancel construct
TYPE_PARSER("CANCEL"_tok >>
    construct<OpenMPCancelConstruct>(
        ("PARALLEL"_tok >> pure(OmpCancelType::Type::Parallel) ||
            "SECTIONS"_tok >> pure(OmpCancelType::Type::Sections) ||
            "DO"_tok >> pure(OmpCancelType::Type::Do) ||
            "TASKGROUP"_tok >> pure(OmpCancelType::Type::Taskgroup)),
        maybe("IF"_tok >> parenthesized(scalarLogicalExpr))))

// Flush construct
TYPE_PARSER("FLUSH"_tok >> construct<OpenMPFlushConstruct>(
                               maybe(parenthesized(Parser<OmpObjectList>{}))))

// Standalone directives
TYPE_PARSER("TARGET ENTER DATA"_tok >>
        construct<OmpStandaloneDirective>(
            construct<OmpStandaloneDirective::TargetEnterData>()) ||
    "TARGET EXIT DATA"_tok >>
        construct<OmpStandaloneDirective>(
            construct<OmpStandaloneDirective::TargetExitData>()) ||
    "TARGET UPDATE"_tok >>
        construct<OmpStandaloneDirective>(
            construct<OmpStandaloneDirective::TargetUpdate>()))

// Directives enclosing structured-block
TYPE_PARSER("MASTER"_tok >>
        construct<OmpBlockDirective>(construct<OmpBlockDirective::Master>()) ||
    "ORDERED"_tok >>
        construct<OmpBlockDirective>(construct<OmpBlockDirective::Ordered>()) ||
    "PARALLEL WORKSHARE"_tok >>
        construct<OmpBlockDirective>(
            construct<OmpBlockDirective::ParallelWorkshare>()) ||
    "PARALLEL"_tok >> construct<OmpBlockDirective>(
                          construct<OmpBlockDirective::Parallel>()) ||
    "TARGET DATA"_tok >> construct<OmpBlockDirective>(
                             construct<OmpBlockDirective::TargetData>()) ||
    "TARGET PARALLEL"_tok >>
        construct<OmpBlockDirective>(
            construct<OmpBlockDirective::TargetParallel>()) ||
    "TARGET TEAMS"_tok >> construct<OmpBlockDirective>(
                              construct<OmpBlockDirective::TargetTeams>()) ||
    "TARGET"_tok >>
        construct<OmpBlockDirective>(construct<OmpBlockDirective::Target>()) ||
    "TASKGROUP"_tok >> construct<OmpBlockDirective>(
                           construct<OmpBlockDirective::Taskgroup>()) ||
    "TASK"_tok >>
        construct<OmpBlockDirective>(construct<OmpBlockDirective::Task>()) ||
    "TEAMS"_tok >>
        construct<OmpBlockDirective>(construct<OmpBlockDirective::Teams>()))

TYPE_PARSER(construct<OmpReductionInitializerClause>("INITIALIZER"_tok >>
    parenthesized("OMP_PRIV"_tok >> "="_tok >> indirect(expr))))

// Declare Reduction Construct
TYPE_PARSER(construct<OpenMPDeclareReductionConstruct>(
    "("_tok >> Parser<OmpReductionOperator>{} / ":"_tok,
    nonemptyList(Parser<DeclarationTypeSpec>{}) / ":"_tok,
    Parser<OmpReductionCombiner>{} / ")"_tok,
    maybe(Parser<OmpReductionInitializerClause>{})))

// declare-target-map-type
TYPE_PARSER(construct<OmpDeclareTargetMapType>(
    "LINK" >> pure(OmpDeclareTargetMapType::Type::Link) ||
    "TO" >> pure(OmpDeclareTargetMapType::Type::To)))

// Declarative directives
TYPE_PARSER(construct<OpenMPDeclareTargetConstruct>(
    construct<OpenMPDeclareTargetConstruct>(construct<OpenMPDeclareTargetConstruct::WithClause>(
        Parser<OmpDeclareTargetMapType>{},
        parenthesized(Parser<OmpObjectList>{}))) ||
    lookAhead(endOfLine) >>
        construct<OpenMPDeclareTargetConstruct>(construct<OpenMPDeclareTargetConstruct::Implicit>()) ||
    construct<OpenMPDeclareTargetConstruct>(
        parenthesized(construct<OpenMPDeclareTargetConstruct::WithExtendedList>(
            Parser<OmpObjectList>{})))))

TYPE_PARSER(construct<OmpReductionCombiner>(Parser<AssignmentStmt>{}) ||
    construct<OmpReductionCombiner>(
        construct<OmpReductionCombiner::FunctionCombiner>(
            construct<Call>(Parser<ProcedureDesignator>{},
                parenthesized(optionalList(actualArgSpec))))))

// OMP END ATOMIC
TYPE_PARSER(construct<OmpEndAtomic>("!$OMP "_sptok >> "END ATOMIC"_tok))

// OMP [SEQ_CST] ATOMIC READ [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicRead>(
    maybe(
        "SEQ_CST"_tok >> construct<OmpAtomicRead::SeqCst1>() / maybe(","_tok)),
    "READ" >> maybe(","_tok) >>
        maybe("SEQ_CST"_tok >> construct<OmpAtomicRead::SeqCst2>()) /
            endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [SEQ_CST] CAPTURE [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicCapture>(
    maybe("SEQ_CST"_tok >>
        construct<OmpAtomicCapture::SeqCst1>() / maybe(","_tok)),
    "CAPTURE" >> maybe(","_tok) >>
        maybe("SEQ_CST"_tok >> construct<OmpAtomicCapture::SeqCst2>()) /
            endOmpLine,
    statement(assignmentStmt), statement(assignmentStmt),
    Parser<OmpEndAtomic>{} / endOmpLine))

// OMP ATOMIC [SEQ_CST] UPDATE [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicUpdate>(
    maybe("SEQ_CST"_tok >>
        construct<OmpAtomicUpdate::SeqCst1>() / maybe(","_tok)),
    "UPDATE" >> maybe(","_tok) >>
        maybe("SEQ_CST"_tok >> construct<OmpAtomicUpdate::SeqCst2>()) /
            endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [SEQ_CST]
TYPE_PARSER(construct<OmpAtomic>(
    maybe("SEQ_CST"_tok >> construct<OmpAtomic::SeqCst>()) / endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// ATOMIC [SEQ_CST] WRITE [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicWrite>(
    maybe(
        "SEQ_CST"_tok >> construct<OmpAtomicWrite::SeqCst1>() / maybe(","_tok)),
    "WRITE" >> maybe(","_tok) >>
        maybe("SEQ_CST"_tok >> construct<OmpAtomicWrite::SeqCst2>()) /
            endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// Atomic Construct
TYPE_PARSER("ATOMIC" >>
    (construct<OpenMPAtomicConstruct>(Parser<OmpAtomicRead>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomicCapture>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomicWrite>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomicUpdate>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomic>{})))

// OMP CRITICAL
TYPE_PARSER("!$OMP "_sptok >> "END"_tok >> "CRITICAL"_tok >>
    construct<OmpEndCritical>(maybe(parenthesized(name))))

TYPE_PARSER("CRITICAL" >>
    construct<OpenMPCriticalConstruct>(maybe(parenthesized(name)),
        maybe("HINT"_tok >> construct<OpenMPCriticalConstruct::Hint>(
                                parenthesized(constantExpr))),
        block, Parser<OmpEndCritical>{}))

// Declare Simd construct
TYPE_PARSER(construct<OpenMPDeclareSimdConstruct>(maybe(parenthesized(name)), Parser<OmpClauseList>{}))

// Declarative construct & Threadprivate directive
TYPE_PARSER(
    lookAhead(!"!$OMP END"_tok) >> "!$OMP "_tok >>
(
"DECLARE REDUCTION" >>
        construct<OpenMPDeclarativeConstruct>(construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclareReductionConstruct>{})) / endOmpLine ||
    "DECLARE SIMD" >> construct<OpenMPDeclarativeConstruct>(
                              Parser<OpenMPDeclareSimdConstruct>{}) / endOmpLine ||
    "DECLARE TARGET" >>
        construct<OpenMPDeclarativeConstruct>(
            construct<OpenMPDeclarativeConstruct>(Parser<OpenMPDeclareTargetConstruct>{})) /endOmpLine ||
    "THREADPRIVATE" >> construct<OpenMPDeclarativeConstruct>(
                           construct<OpenMPDeclarativeConstruct::Threadprivate>(
                               parenthesized(Parser<OmpObjectList>{}))/ endOmpLine)))

// Loop Construct
TYPE_PARSER(construct<OpenMPLoopConstruct>(Parser<OmpLoopDirective>{},
    Parser<OmpClauseList>{} / endOmpLine, Parser<DoConstruct>{},
    maybe(Parser<OmpEndLoopDirective>{} / endOmpLine)))

// Block Construct
TYPE_PARSER(construct<OpenMPBlockConstruct>(Parser<OmpBlockDirective>{},
    Parser<OmpClauseList>{} / endOmpLine, block,
    Parser<OmpEndBlockDirective>{} / endOmpLine))

TYPE_PARSER(construct<OpenMPStandaloneConstruct>(
    Parser<OmpStandaloneDirective>{}, Parser<OmpClauseList>{} / endOmpLine))


// OMP BARRIER
TYPE_PARSER("BARRIER"_tok >> construct<OpenMPBarrierConstruct>() / endOmpLine)

// OMP TASKWAIT
TYPE_PARSER("TASKWAIT"_tok >> construct<OpenMPTaskwaitConstruct>() / endOmpLine)

// OMP TASKYIELD
TYPE_PARSER(
    "TASKYIELD"_tok >> construct<OpenMPTaskyieldConstruct>() / endOmpLine)

// OMP SINGLE
TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    construct<OmpEndSingle>("SINGLE"_tok >> Parser<OmpClauseList>{}))

TYPE_PARSER("SINGLE"_tok >>
    construct<OpenMPSingleConstruct>(Parser<OmpClauseList>{} / endOmpLine,
        block, Parser<OmpEndSingle>{} / endOmpLine))

TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    construct<OmpEndWorkshare>("WORKSHARE"_tok))

// OMP WORKSHARE
TYPE_PARSER("WORKSHARE"_tok >>
    construct<OpenMPWorkshareConstruct>(endOmpLine >> block,
        Parser<OmpEndWorkshare>{} >>
            maybe(construct<OmpNowait>("NOWAIT"_tok)) / endOmpLine))

// OMP END DO SIMD [NOWAIT]
TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    construct<OmpEndDoSimd>(
        "DO SIMD"_tok >> maybe(construct<OmpNowait>("NOWAIT"_tok))))

// OMP DO SIMD
TYPE_PARSER("DO SIMD"_tok >>
    construct<OpenMPDoSimdConstruct>(Parser<OmpClauseList>{} / endOmpLine,
        Parser<DoConstruct>{}, maybe(Parser<OmpEndDoSimd>{})))

// OMP END DO [NOWAIT]
TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    construct<OmpEndDo>("DO"_tok >> maybe(construct<OmpNowait>("NOWAIT"_tok))))

// OMP DO
TYPE_PARSER("DO"_tok >>
    construct<OpenMPDoConstruct>(Parser<OmpClauseList>{} / endOmpLine,
        Parser<DoConstruct>{}, maybe(Parser<OmpEndDo>{})))

// OMP END SECTIONS [NOWAIT]
TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    "SECTIONS"_tok >>
    construct<OmpEndSections>(
        maybe("NOWAIT"_tok >> construct<OmpNowait>()) / endOmpLine))

// OMP SECTIONS
TYPE_PARSER("SECTIONS"_tok >>
    construct<OpenMPSectionsConstruct>(Parser<OmpClauseList>{} / endOmpLine,
        block, Parser<OmpEndSections>{}))

// OMP END PARALLEL SECTIONS [NOWAIT]
TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    "PARALLEL SECTIONS"_tok >>
    construct<OmpEndParallelSections>(
        maybe("NOWAIT"_tok >> construct<OmpNowait>()) / endOmpLine))

// OMP PARALLEL SECTIONS
TYPE_PARSER("PARALLEL SECTIONS"_tok >>
    construct<OpenMPParallelSectionsConstruct>(
        Parser<OmpClauseList>{} / endOmpLine, block,
        Parser<OmpEndParallelSections>{}))

TYPE_CONTEXT_PARSER("OpenMP construct"_en_US,
    skipEmptyLines >> space >> "!$OMP "_sptok >> lookAhead(!"END"_tok) >>
        (construct<OpenMPConstruct>(
             indirect(Parser<OpenMPStandaloneConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPBarrierConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPTaskwaitConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPTaskyieldConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPSingleConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPDoSimdConstruct>{})) ||
            construct<OpenMPConstruct>(indirect(Parser<OpenMPDoConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPSectionsConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPParallelSectionsConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPWorkshareConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPLoopConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPBlockConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPAtomicConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPCriticalConstruct>{})) ||
            construct<OpenMPConstruct>(
                indirect(Parser<OpenMPFlushConstruct>{})) ||
            "SECTION"_tok >> endOmpLine >>
                construct<OpenMPConstruct>(construct<OmpSection>())))

// End Omp directives
TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    construct<OmpEndBlockDirective>(indirect(Parser<OmpBlockDirective>{})))

TYPE_PARSER(skipEmptyLines >> space >> "!$OMP "_sptok >> "END"_tok >>
    construct<OmpEndLoopDirective>(indirect(Parser<OmpLoopDirective>{})))

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_OPENMP_GRAMMAR_H_
