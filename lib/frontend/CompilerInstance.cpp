//===--- CompilerInstance.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"

#include "flang/Frontend/FrontendAction.h"
#include "flang/Frontend/FrontendDiagnostic.h"
#include "flang/Frontend/LogDiagnosticPrinter.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "flang/Frontend/Utils.h"

#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Stack.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/stat.h>
#include <system_error>
#include <time.h>
#include <utility>

using namespace clang;

CompilerInstance::CompilerInstance()
    : Invocation(new CompilerInvocation()) {}

CompilerInstance::~CompilerInstance() {
  assert(OutputFiles.empty() && "Still output files in flight?");
}

void CompilerInstance::setInvocation(
    std::shared_ptr<CompilerInvocation> Value) {
  Invocation = std::move(Value);
}

void CompilerInstance::setDiagnostics(DiagnosticsEngine *Value) {
  Diagnostics = Value;
}

void CompilerInstance::setTarget(TargetInfo *Value) { Target = Value; }
void CompilerInstance::setAuxTarget(TargetInfo *Value) { AuxTarget = Value; }

void CompilerInstance::setFileManager(FileManager *Value) {
  FileMgr = Value;
}

void CompilerInstance::setSourceManager(SourceManager *Value) {
  SourceMgr = Value;
}

void CompilerInstance::createDiagnostics(DiagnosticConsumer *Client,
                                         bool ShouldOwnClient) {
  Diagnostics = createDiagnostics(&getDiagnosticOpts(), Client,
                                  ShouldOwnClient);
}

IntrusiveRefCntPtr<DiagnosticsEngine>
CompilerInstance::createDiagnostics(DiagnosticOptions *Opts,
                                    DiagnosticConsumer *Client,
                                    bool ShouldOwnClient,
                                    const CodeGenOptions *CodeGenOpts) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticsEngine>
      Diags(new DiagnosticsEngine(DiagID, Opts));

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  if (Client) {
    Diags->setClient(Client, ShouldOwnClient);
  } else
    Diags->setClient(new TextDiagnosticPrinter(llvm::errs(), Opts));

  // Configure our handling of diagnostics.
  ProcessWarningOptions(*Diags, *Opts);

  return Diags;
}

// File Manager

FileManager *CompilerInstance::createFileManager(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  if (!VFS)
    VFS = FileMgr ? &FileMgr->getVirtualFileSystem()
                  : createVFSFromCompilerInvocation(getInvocation(),
                                                    getDiagnostics());
  assert(VFS && "FileManager has no VFS?");
  FileMgr = new FileManager(getFileSystemOpts(), std::move(VFS));
  return FileMgr.get();
}

// Source Manager

void CompilerInstance::createSourceManager(FileManager &FileMgr) {
  SourceMgr = new SourceManager(getDiagnostics(), FileMgr);
}

// Output Files

void CompilerInstance::addOutputFile(OutputFile &&OutFile) {
  OutputFiles.push_back(std::move(OutFile));
}

void CompilerInstance::clearOutputFiles(bool EraseFiles) {
  for (OutputFile &OF : OutputFiles) {
    if (!OF.TempFilename.empty()) {
      if (EraseFiles) {
        llvm::sys::fs::remove(OF.TempFilename);
      } else {
        SmallString<128> NewOutFile(OF.Filename);

        // If '-working-directory' was passed, the output filename should be
        // relative to that.
        FileMgr->FixupRelativePath(NewOutFile);
        if (std::error_code ec =
                llvm::sys::fs::rename(OF.TempFilename, NewOutFile)) {
          getDiagnostics().Report(diag::err_unable_to_rename_temp)
            << OF.TempFilename << OF.Filename << ec.message();

          llvm::sys::fs::remove(OF.TempFilename);
        }
      }
    } else if (!OF.Filename.empty() && EraseFiles)
      llvm::sys::fs::remove(OF.Filename);
  }
  OutputFiles.clear();
  NonSeekStream.reset();
}

std::unique_ptr<raw_pwrite_stream>
CompilerInstance::createDefaultOutputFile(bool Binary, StringRef InFile,
                                          StringRef Extension) {
  return createOutputFile(getFrontendOpts().OutputFile, Binary,
                          /*RemoveFileOnSignal=*/true, InFile, Extension,
                          /*UseTemporary=*/true);
}

std::unique_ptr<raw_pwrite_stream> CompilerInstance::createNullOutputFile() {
  return std::make_unique<llvm::raw_null_ostream>();
}

std::unique_ptr<raw_pwrite_stream>
CompilerInstance::createOutputFile(StringRef OutputPath, bool Binary,
                                   bool RemoveFileOnSignal, StringRef InFile,
                                   StringRef Extension, bool UseTemporary,
                                   bool CreateMissingDirectories) {
  std::string OutputPathName, TempPathName;
  std::error_code EC;
  std::unique_ptr<raw_pwrite_stream> OS = createOutputFile(
      OutputPath, EC, Binary, RemoveFileOnSignal, InFile, Extension,
      UseTemporary, CreateMissingDirectories, &OutputPathName, &TempPathName);
  if (!OS) {
    getDiagnostics().Report(diag::err_fe_unable_to_open_output) << OutputPath
                                                                << EC.message();
    return nullptr;
  }

  // Add the output file -- but don't try to remove "-", since this means we are
  // using stdin.
  addOutputFile(
      OutputFile((OutputPathName != "-") ? OutputPathName : "", TempPathName));

  return OS;
}

std::unique_ptr<llvm::raw_pwrite_stream> CompilerInstance::createOutputFile(
    StringRef OutputPath, std::error_code &Error, bool Binary,
    bool RemoveFileOnSignal, StringRef InFile, StringRef Extension,
    bool UseTemporary, bool CreateMissingDirectories,
    std::string *ResultPathName, std::string *TempPathName) {
  assert((!CreateMissingDirectories || UseTemporary) &&
         "CreateMissingDirectories is only allowed when using temporary files");

  std::string OutFile, TempFile;
  if (!OutputPath.empty()) {
    OutFile = OutputPath;
  } else if (InFile == "-") {
    OutFile = "-";
  } else if (!Extension.empty()) {
    SmallString<128> Path(InFile);
    llvm::sys::path::replace_extension(Path, Extension);
    OutFile = Path.str();
  } else {
    OutFile = "-";
  }

  std::unique_ptr<llvm::raw_fd_ostream> OS;
  std::string OSFile;

  if (UseTemporary) {
    if (OutFile == "-")
      UseTemporary = false;
    else {
      llvm::sys::fs::file_status Status;
      llvm::sys::fs::status(OutputPath, Status);
      if (llvm::sys::fs::exists(Status)) {
        // Fail early if we can't write to the final destination.
        if (!llvm::sys::fs::can_write(OutputPath)) {
          Error = make_error_code(llvm::errc::operation_not_permitted);
          return nullptr;
        }

        // Don't use a temporary if the output is a special file. This handles
        // things like '-o /dev/null'
        if (!llvm::sys::fs::is_regular_file(Status))
          UseTemporary = false;
      }
    }
  }

  if (UseTemporary) {
    // Create a temporary file.
    // Insert -%%%%%%%% before the extension (if any), and because some tools
    // (noticeable, clang's own GlobalModuleIndex.cpp) glob for build
    // artifacts, also append .tmp.
    StringRef OutputExtension = llvm::sys::path::extension(OutFile);
    SmallString<128> TempPath =
        StringRef(OutFile).drop_back(OutputExtension.size());
    TempPath += "-%%%%%%%%";
    TempPath += OutputExtension;
    TempPath += ".tmp";
    int fd;
    std::error_code EC =
        llvm::sys::fs::createUniqueFile(TempPath, fd, TempPath);

    if (CreateMissingDirectories &&
        EC == llvm::errc::no_such_file_or_directory) {
      StringRef Parent = llvm::sys::path::parent_path(OutputPath);
      EC = llvm::sys::fs::create_directories(Parent);
      if (!EC) {
        EC = llvm::sys::fs::createUniqueFile(TempPath, fd, TempPath);
      }
    }

    if (!EC) {
      OS.reset(new llvm::raw_fd_ostream(fd, /*shouldClose=*/true));
      OSFile = TempFile = TempPath.str();
    }
    // If we failed to create the temporary, fallback to writing to the file
    // directly. This handles the corner case where we cannot write to the
    // directory, but can write to the file.
  }

  if (!OS) {
    OSFile = OutFile;
    OS.reset(new llvm::raw_fd_ostream(
        OSFile, Error,
        (Binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text)));
    if (Error)
      return nullptr;
  }

  // Make sure the out stream file gets removed if we crash.
  if (RemoveFileOnSignal)
    llvm::sys::RemoveFileOnSignal(OSFile);

  if (ResultPathName)
    *ResultPathName = OutFile;
  if (TempPathName)
    *TempPathName = TempFile;

  if (!Binary || OS->supportsSeeking())
    return std::move(OS);

  auto B = std::make_unique<llvm::buffer_ostream>(*OS);
  assert(!NonSeekStream);
  NonSeekStream = std::move(OS);
  return std::move(B);
}

// Initialization Utilities

bool CompilerInstance::InitializeSourceManager(const FrontendInputFile &Input){
  return InitializeSourceManager(
      Input, getDiagnostics(), getFileManager(), getSourceManager(), getFrontendOpts());
}

// static
bool CompilerInstance::InitializeSourceManager(
    const FrontendInputFile &Input, DiagnosticsEngine &Diags,
    FileManager &FileMgr, SourceManager &SourceMgr, const FrontendOptions &Opts) {

  // TODO(peterwaller-arm): SourceManager.
  SrcMgr::CharacteristicKind Kind = SrcMgr::C_User;

  if (Input.isBuffer()) {
    SourceMgr.setMainFileID(SourceMgr.createFileID(SourceManager::Unowned,
                                                   Input.getBuffer(), Kind));
    assert(SourceMgr.getMainFileID().isValid() &&
           "Couldn't establish MainFileID!");
    return true;
  }

  StringRef InputFile = Input.getFile();

  // Figure out where to get and map in the main file.
  if (InputFile != "-") {
    auto FileOrErr = FileMgr.getFileRef(InputFile, /*OpenFile=*/true);
    if (!FileOrErr) {
      // FIXME: include the error in the diagnostic.
      consumeError(FileOrErr.takeError());
      Diags.Report(diag::err_fe_error_reading) << InputFile;
      return false;
    }
    FileEntryRef File = *FileOrErr;

    // The natural SourceManager infrastructure can't currently handle named
    // pipes, but we would at least like to accept them for the main
    // file. Detect them here, read them with the volatile flag so FileMgr will
    // pick up the correct size, and simply override their contents as we do for
    // STDIN.
    if (File.getFileEntry().isNamedPipe()) {
      auto MB =
          FileMgr.getBufferForFile(&File.getFileEntry(), /*isVolatile=*/true);
      if (MB) {
        // Create a new virtual file that will have the correct size.
        const FileEntry *FE =
            FileMgr.getVirtualFile(InputFile, (*MB)->getBufferSize(), 0);
        SourceMgr.overrideFileContents(FE, std::move(*MB));
        SourceMgr.setMainFileID(
            SourceMgr.createFileID(FE, SourceLocation(), Kind));
      } else {
        Diags.Report(diag::err_cannot_open_file) << InputFile
                                                 << MB.getError().message();
        return false;
      }
    } else {
      SourceMgr.setMainFileID(
          SourceMgr.createFileID(File, SourceLocation(), Kind));
    }
  } else {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> SBOrErr =
        llvm::MemoryBuffer::getSTDIN();
    if (std::error_code EC = SBOrErr.getError()) {
      Diags.Report(diag::err_fe_error_reading_stdin) << EC.message();
      return false;
    }
    std::unique_ptr<llvm::MemoryBuffer> SB = std::move(SBOrErr.get());

    const FileEntry *File = FileMgr.getVirtualFile(SB->getBufferIdentifier(),
                                                   SB->getBufferSize(), 0);
    SourceMgr.setMainFileID(
        SourceMgr.createFileID(File, SourceLocation(), Kind));
    SourceMgr.overrideFileContents(File, std::move(SB));
  }

  assert(SourceMgr.getMainFileID().isValid() &&
         "Couldn't establish MainFileID!");
  return true;
}

// High-Level Operations

bool CompilerInstance::ExecuteAction(FrontendAction &Act) {
  assert(hasDiagnostics() && "Diagnostics engine is not initialized!");
  assert(!getFrontendOpts().ShowHelp && "Client must handle '-help'!");
  assert(!getFrontendOpts().ShowVersion && "Client must handle '-version'!");

  // Mark this point as the bottom of the stack if we don't have somewhere
  // better. We generally expect frontend actions to be invoked with (nearly)
  // DesiredStackSpace available.
  noteBottomOfStack();

  if (!Act.PrepareToExecute(*this))
    return false;

  // Create the target instance.
  setTarget(TargetInfo::CreateTargetInfo(getDiagnostics(),
                                         getInvocation().TargetOpts));
  if (!hasTarget())
    return false;

  for (const FrontendInputFile &FIF : getFrontendOpts().Inputs) {
    // Reset the ID tables if we are reusing the SourceManager and parsing
    // regular files.
    if (hasSourceManager() && !Act.isModelParsingAction())
      getSourceManager().clearIDTables();

    if (Act.BeginSourceFile(*this, FIF)) {
      if (llvm::Error Err = Act.Execute()) {
        consumeError(std::move(Err)); // FIXME this drops errors on the floor.
      }
      Act.EndSourceFile();
    }
  }

  // Notify the diagnostic client that all files were processed.
  getDiagnostics().getClient()->finish();

  return !getDiagnostics().getClient()->getNumErrors();
}

