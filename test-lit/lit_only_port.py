""" lit_only_port.py

    Ports ctest style F18 tests to be compatible with lit

    Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
    See https://llvm.org/LICENSE.txt for license information.
    SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import argparse
import os
import re
import shlex
import sys
import subprocess
from pathlib import Path

from test_splitter import TestSplitter

F18_BIN = "~/f18/build/install/bin"
F18_TEST = "~/f18/build-f18/test-lit"

ROOT = Path.cwd()
while ROOT.name != "f18":
    ROOT = ROOT.parent
TEMPLATE = "!RUN: %test_error %s %flang"
FAILS = []
TS = TestSplitter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs='+',
        help="Run the porter on (a) test(s) or test suite(s)")
    parser.add_argument(
        "--output", '-o',
        help="Save the ported file(s) in the specified directory. \
        If no directory is specifed, the program will try to find \
        an appropriate output directory.")
    parser.add_argument(
        "--clean", "-c", action='store_true',
        help="Remove old tests from output directory"
    )
    parser.add_argument(
        "--glob", "-g", action='store_true',
        help="Glob files. Default behaviour will only get files in the \
        immediate input path"
    )
    parser.add_argument(
        "--error", "-e", action='store_true',
        help="Will only port error tests"
    )
    args = parser.parse_args()

    output = ""
    if args.output:
        output = Path(args.output)
        if output.exists():
            if not output.is_dir():
                print(
                    "Output path should be a directory or not set",
                    file=sys.stderr)
                exit(1)
        else:
            try:
                output.mkdir(parents=True)
            except PermissionError:
                print(
                    "Invalid permission to create directory at given path",
                    file=sys.stderr)
                exit(1)

    tests = []
    for filename in args.input:
        path = Path(filename)
        if path.is_dir():
            if args.glob:
                tests += [f for f in path.glob('**/*') if f.is_file()]
            else:
                tests += [f for f in path.iterdir() if f.is_file()]
        elif path.is_file():
            tests.append(path)

    if args.error:
        tests = [test for test in tests if test.name in TS.error_tests]
    else:
        tests = [test for test in tests if test.name in TS.tests]
    if not tests:
        print("No tests found", file=sys.stderr)
        exit(1)
    else:
        for test in tests:
            if not output:
                dirname = test.parent.name
                output = ROOT.joinpath("test-lit", dirname)
                if not output.exists():
                    try:
                        output.mkdir(parents=True)
                    except PermissionError:
                        print(
                            "Invalid permission to create directory at given path",
                            file=sys.stderr)
                        exit(1)

            if args.clean:
                suffixes = ['.f', '.F']
                for suffix in suffixes:
                    if list(output.glob('**/*{}*'.format(suffix))) != 0:
                        for old_test in TS.tests:
                            path = output.joinpath(old_test)
                            if path.exists():
                                path.unlink()

            port_single_test(test, output)
            cleanup()
    if FAILS:
        for fail in FAILS:
            print("Test {} failed".format(fail))
    else:
        print("No fails detected")


def port_single_test(filename, output):
    test = filename.name
    new_path = output.joinpath(test)
    rel_path = output.resolve().relative_to(ROOT)
    print("Porting {} to {}".format(test, rel_path))
    if test in TS.error_tests:
        port_error_test(filename, new_path)
        #if test not in FAILS:
        #    test_single_port(new_path, filename)
    else:
        print(
            "{} could not be ported as it is not supported".format(test),
            file=sys.stderr)
        FAILS.append(test)


def port_error_test(filename, savepath):
    test = filename.name
    lines = []
    try:
        with filename.open() as read_file:
            lines = read_file.readlines()
            prefixes = ["ERROR", "WARNING"]
            index = -1
            # Add run line to untouched test
            for line in lines:
                if any(prefix in line for prefix in prefixes):
                    index = lines.index(line) - 1
                    if index < 0:
                        index = 0
                    break
                # Find the first line without comment character
                if line != "\n" and not line.startswith("!"):
                    index = lines.index(line)
                    break
            lines.insert(index, "\n\n")
            lines.insert(
                index, TEMPLATE)

            try:
                with savepath.open('w') as write_file:
                    write_file.writelines(lines)
                print("{} completed".format(test))
            except IOError as e:
                print(
                    "Could not write to {} because {}".format(
                        savepath, e.strerror), file=sys.stderr)
                FAILS.append(test)
    except IOError as e:
        print(
            "Could not open {} because {}".format(
                filename, e.strerror), file=sys.stderr)
        FAILS.append(test)


def run(command):
    command = "PATH={}:$PATH;{}".format(F18_BIN, command)
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True).stdout.decode('utf-8')


def test_single_port(new_path, old_path):
    test = new_path.name
    new_path = Path(F18_TEST).joinpath(new_path.parent.name, new_path.name
                                       )
    print("Testing {}".format(test))
    old_result = run(
        "{}/test_errors.sh {} $(which flang) 2>&1".format(
            old_path.parent, test
        )
    )
    new_result = run(
        "lit {} -a".format(new_path)
    )
    if "PASS" not in new_result:
        FAILS.append(test)
    pattern = re.compile(r".*{}.*".format(test))
    old_match = re.search(pattern, old_result)
    new_match = re.search(pattern, new_result)
    if old_match and new_match:
        print("Matches")
    else:
        print(old_result, new_result)
    if ("PASS" in old_result) == ("PASS" in new_result):
        print("Success")
    else:
        print("Failure")
        print("OLD RESULT")
        print("=" * 20)
        print(old_result)
        print("NEW RESULT")
        print("=" * 20)
        print(new_result)


def cleanup():
    files = [
        f for f in Path.cwd().iterdir()
        if f.is_file() and f.name.endswith("mod")
    ]
    for filename in files:
        filename.unlink()


main()
