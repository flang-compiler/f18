#include "clang/Basic/Stack.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"

extern int cc1_main(llvm::ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr);

void dummy() {}

int main(int argc_, const char* argv_[]) {
  clang::noteBottomOfStack();
  llvm::InitLLVM X(argc_, argv_);
  llvm::SmallVector<const char *, 256> argv(argv_, argv_ + argc_);
  llvm::ArrayRef<const char *> argvRef = argv;
  llvm::InitializeAllTargets();
  return cc1_main(argvRef.slice(1), argv[0], (void *)(intptr_t)dummy);
}