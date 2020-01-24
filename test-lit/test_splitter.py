""" test_splitter.py

    Aids lit_port.py in finding tests from the CMakeLists.txt specifying tests.

    Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
    See https://llvm.org/LICENSE.txt for license information.
    SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
    
"""

import argparse
import re
from pathlib import Path


class TestSplitter:
    def __init__(self):
        self.error_tests = []
        self.modfile_tests = []
        self.symbol_tests = []
        self.generic_tests = []
        self.folding_tests = []
        self.unsupported_tests = []
        self.test_types = ['ERROR', 'MODFILE', 'SYMBOL', 'GENERIC', 'FOLDING']
        root = Path(__file__).resolve()
        while root.name != "f18":
            root = root.parent
        self.root = root
        self.test_path = self.root / "test"
        self.get_semantics_tests()
        self.get_folding_tests()
        self.tests = (self.error_tests + self.modfile_tests +
                      self.symbol_tests + self.generic_tests +
                      self.folding_tests +
                      self.unsupported_tests)

    def get_semantics_tests(self):
        semantics_path = self.test_path / "semantics"
        semantics_cmake_path = semantics_path / "CMakeLists.txt"
        with semantics_cmake_path.open() as read_file:
            current_test_set = None
            for line in read_file.readlines():
                match = re.match(r"^set\((.*)$", line)
                if match:
                    if match.group(1) == "ERROR_TESTS":
                        current_test_set = self.error_tests
                    elif match.group(1) == "MODFILE_TESTS":
                        current_test_set = self.modfile_tests
                    elif match.group(1) == "SYMBOL_TESTS":
                        current_test_set = self.symbol_tests
                    else:
                        current_test_set = self.generic_tests
                elif re.match(r".*\..*f[90|\]| \n]", line):
                    if current_test_set is None:
                        continue
                    if line.strip().startswith("#"):
                        match = re.match(r".*#\s*(.*\.f.*)", line.lower())
                        if match:
                            self.unsupported_tests.append(match.group(1))
                            continue
                    if "*" in line:
                        path = Path(semantics_path)
                        pattern = line.strip().split("*")[0] + "*"
                        for test in list(path.glob(pattern)):
                            current_test_set.append(test.name)
                    else:
                        current_test_set.append(line.strip())

    def get_folding_tests(self):
        evaluate_path = self.test_path / "evaluate"
        evaluate_cmake_path = evaluate_path / "CMakeLists.txt"
        with evaluate_cmake_path.open() as read_file:
            for line in read_file.readlines():
                if line.strip().endswith(".f90"):
                    self.folding_tests.append(line.strip())


def main():
    ts = TestSplitter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error", "-e", help="Display error tests", action="store_true")
    parser.add_argument(
        "--symbol", "-s", help="Display symbol tests", action="store_true")
    parser.add_argument(
        "--modfile", "-m", help="Display modfile tests", action="store_true")
    parser.add_argument(
        "--generic", "-g", help="Display generic tests", action="store_true")
    parser.add_argument(
        "--folding", "-f", help="Display folding tests", action="store_true")
    parser.add_argument(
        "--unsupported", "-u", help="Display unsupported tests", action="store_true")
    parser.add_argument(
        "--all", "-a", help="Display all tests", action="store_true")
    parser.add_argument(
        "--info", "-i", help="Display info about tests", action="store_true")
    args = parser.parse_args()
    if args.info:
        print(
            """
Semantics has:
{} Error Tests
{} Symbol Tests
{} Modfile Tests
{} Generic Tests

Evaluate has:
{} Folding Tests
""".format(
                len(ts.error_tests), len(ts.symbol_tests),
                len(ts.modfile_tests), len(ts.generic_tests),
                len(ts.folding_tests)
            )
        )
    if args.error:
        for test in ts.error_tests:
            print(test)
    if args.symbol:
        for test in ts.symbol_tests:
            print(test)
    if args.modfile:
        for test in ts.modfile_tests:
            print(test)
    if args.generic:
        for test in ts.generic_tests:
            print(test)
    if args.folding:
        for test in ts.folding_tests:
            print(test)
    if args.unsupported:
        for test in ts.unsupported_tests:
            print(test)
    elif args.all:
        for test in ts.error_tests:
            print(test + " (ERROR)")
        for test in ts.symbol_tests:
            print(test + " (SYMBOL)")
        for test in ts.modfile_tests:
            print(test + " (MODFILE)")
        for test in ts.generic_tests:
            print(test + " (GENERIC)")
        for test in ts.folding_tests:
            print(test + " (FOLDING)")
        for test in ts.unsupported_tests:
            print(test + " (UNSUPPORTED)")


if __name__ == '__main__':
    main()
