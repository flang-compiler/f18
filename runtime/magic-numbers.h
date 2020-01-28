#if 0 /*===-- runtime/magic-numbers.h -----------------------------------===*/
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===--------------------------------------------------------------------===*/
#endif
#if 0
This header can be included into both Fortran and C.

This file defines various code values that need to be exported
to predefined Fortran standard modules as well as to C/C++
code in the compiler and runtime library.
These include:
 - the error/end code values that can be returned
   to an IOSTAT= or STAT= specifier on a Fortran I/O statement
   or coindexed data reference (see Fortran 2018 12.11.5,
   16.10.2, and 16.10.2.33)
Codes from <errno.h>, e.g. ENOENT, are assumed to be positive
and are used "raw" as IOSTAT values.
#endif
#ifndef FORTRAN_RUNTIME_MAGIC_NUMBERS_H_
#define FORTRAN_RUNTIME_MAGIC_NUMBERS_H_

#define FORTRAN_RUNTIME_IOSTAT_END (-1)
#define FORTRAN_RUNTIME_IOSTAT_EOR (-2)
#define FORTRAN_RUNTIME_IOSTAT_FLUSH (-3)
#define FORTRAN_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT 255

#define FORTRAN_RUNTIME_STAT_FAILED_IMAGE 10
#define FORTRAN_RUNTIME_STAT_LOCKED 11
#define FORTRAN_RUNTIME_STAT_LOCKED_OTHER_IMAGE 12
#define FORTRAN_RUNTIME_STAT_STOPPED_IMAGE 13
#define FORTRAN_RUNTIME_STAT_UNLOCKED 14
#define FORTRAN_RUNTIME_STAT_UNLOCKED_FAILED_IMAGE 15
#endif
