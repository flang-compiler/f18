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

#include "testing.h"
#include "../../include/flang/ISO_Fortran_binding.h"
#include "../../runtime/descriptor.h"
#include <type_traits>
#ifdef VERBOSE
#include <iostream>
#endif

using namespace Fortran::runtime;
using namespace Fortran::ISO;

// CFI_CDESC_T test helpers
template<int rank> class Test_CFI_CDESC_T {
public:
  Test_CFI_CDESC_T() : dvStorage_{0} {};
  ~Test_CFI_CDESC_T(){};
  void Check() {
    // Test CFI_CDESC_T macro defined in section 18.5.4 of F2018 standard
    // CFI_CDESC_T must give storage that is:
    using type = decltype(dvStorage_);
    // unqualified
    MATCH(false, std::is_const<type>::value);
    MATCH(false, std::is_volatile<type>::value);
    // suitable in size
    if (rank > 0) {
      MATCH(sizeof(dvStorage_), Descriptor::SizeInBytes(rank_, false));
    } else {  // C++ implem overalocates for rank=0 by 24bytes.
      MATCH(true, sizeof(dvStorage_) >= Descriptor::SizeInBytes(rank_, false));
    }
    // suitable in alignment
    MATCH(0,
        reinterpret_cast<std::uintptr_t>(&dvStorage_) &
            (alignof(CFI_cdesc_t) - 1));
  }

private:
  static constexpr int rank_ = rank;
  CFI_CDESC_T(rank) dvStorage_;
};

template<int rank> static void TestCdescMacroForAllRanksSmallerThan() {
  static_assert(rank > 0, "rank<0!");
  Test_CFI_CDESC_T<rank> obj;
  obj.Check();
  TestCdescMacroForAllRanksSmallerThan<rank - 1>();
}

template<> void TestCdescMacroForAllRanksSmallerThan<0>() {
  Test_CFI_CDESC_T<0> obj;
  obj.Check();
}

// CFI_establish test helper
static void AddNoiseToCdesc(CFI_cdesc_t *dv, CFI_rank_t rank) {
  static int trap;
  dv->rank = 16;
  dv->base_addr = reinterpret_cast<void *>(&trap);
  dv->elem_len = 320;
  dv->type = CFI_type_struct;
  dv->attribute = CFI_attribute_pointer;
  for (int i{0}; i < rank; i++) {
    dv->dim[i].extent = -42;
    dv->dim[i].lower_bound = -42;
    dv->dim[i].sm = -42;
  }
}

#ifdef VERBOSE
static void DumpTestWorld(const void *bAddr, CFI_attribute_t attr,
    CFI_type_t ty, std::size_t eLen, CFI_rank_t rank,
    const CFI_index_t *eAddr) {
  std::cout << " base_addr: " << std::hex
            << reinterpret_cast<std::intptr_t>(bAddr)
            << " attribute: " << static_cast<int>(attr) << std::dec
            << " type: " << static_cast<int>(ty) << " elem_len: " << eLen
            << " rank: " << static_cast<int>(rank) << " extent: " << std::hex
            << reinterpret_cast<std::intptr_t>(eAddr) << std::endl
            << std::dec;
}
#endif

static void check_CFI_establish(CFI_cdesc_t *dv, void *base_addr,
    CFI_attribute_t attribute, CFI_type_t type, std::size_t elem_len,
    CFI_rank_t rank, const CFI_index_t extents[]) {
#ifdef VERBOSE
  DumpTestWorld(base_addr, attribute, type, elem_len, rank, extent);
#endif
  // CFI_establish reqs from F2018 section 18.5.5
  int retCode =
      CFI_establish(dv, base_addr, attribute, type, elem_len, rank, extents);
  Descriptor *res = reinterpret_cast<Descriptor *>(dv);
  if (retCode == CFI_SUCCESS) {
    res->Check();
    MATCH((attribute == CFI_attribute_pointer), res->IsPointer());
    MATCH((attribute == CFI_attribute_allocatable), res->IsAllocatable());
    MATCH(rank, res->rank());
    MATCH(reinterpret_cast<std::intptr_t>(dv->base_addr),
        reinterpret_cast<std::intptr_t>(base_addr));
    MATCH(true, dv->version == CFI_VERSION);
    if (base_addr != nullptr) {
      MATCH(true, res->IsContiguous());
      for (int i{0}; i < rank; ++i) {
        MATCH(extents[i], res->GetDimension(i).Extent());
      }
    }
    if (attribute == CFI_attribute_allocatable) {
      MATCH(res->IsAllocated(), false);
    }
    if (attribute == CFI_attribute_pointer) {
      if (base_addr != nullptr) {
        for (int i{0}; i < rank; ++i) {
          MATCH(0, res->GetDimension(i).LowerBound());
        }
      }
    }
    if (type == CFI_type_struct || type == CFI_type_char ||
        type == CFI_type_other) {
      MATCH(elem_len, res->ElementBytes());
    }
  }
  // Checking failure/success according to combination of args forbidden by the
  // standard:
  int numErr{0};
  int expectedRetCode{CFI_SUCCESS};
  if (base_addr != nullptr && attribute == CFI_attribute_allocatable) {
    ++numErr;
    expectedRetCode = CFI_ERROR_BASE_ADDR_NOT_NULL;
  }
  if (rank < 0 || rank > CFI_MAX_RANK) {
    ++numErr;
    expectedRetCode = CFI_INVALID_RANK;
  }
  if (type < 0 || type > CFI_type_struct) {
    ++numErr;
    expectedRetCode = CFI_INVALID_TYPE;
  }

  if ((type == CFI_type_struct || type == CFI_type_char ||
          type == CFI_type_other) &&
      elem_len <= 0) {
    ++numErr;
    expectedRetCode = CFI_INVALID_ELEM_LEN;
  }
  if (rank > 0 && base_addr != nullptr && extents == nullptr) {
    ++numErr;
    expectedRetCode = CFI_INVALID_EXTENT;
  }
  if (numErr > 1) {
    MATCH(true, retCode != CFI_SUCCESS);
  } else {
    MATCH(retCode, expectedRetCode);
  }
}

static void run_CFI_establish_tests() {
  // Testing CFI_establish definied in section 18.5.5
  CFI_index_t extents[CFI_MAX_RANK];
  for (int i{0}; i < CFI_MAX_RANK; ++i) {
    extents[i] = i + 66;
  }
  CFI_CDESC_T(CFI_MAX_RANK) dv_storage;
  CFI_cdesc_t *dv{reinterpret_cast<CFI_cdesc_t *>(&dv_storage)};
  static char base;
  void *dummyAddr = reinterpret_cast<void *>(&base);
  // Define test space
  CFI_attribute_t attrCases[]{
      CFI_attribute_pointer, CFI_attribute_allocatable, CFI_attribute_other};
  CFI_type_t typeCases[]{CFI_type_int, CFI_type_struct, CFI_type_double,
      CFI_type_char, CFI_type_other, CFI_type_struct + 1};
  CFI_index_t *extentCases[]{extents, static_cast<CFI_index_t *>(nullptr)};
  void *baseAddrCases[]{dummyAddr, static_cast<void *>(nullptr)};
  CFI_rank_t rankCases[]{0, 1, CFI_MAX_RANK, CFI_MAX_RANK + 1};
  std::size_t lenCases[]{0, 42};

  for (CFI_attribute_t attribute : attrCases) {
    for (void *base_addr : baseAddrCases) {
      for (CFI_index_t *extent : extentCases) {
        for (CFI_rank_t rank : rankCases) {
          for (CFI_type_t type : typeCases) {
            for (size_t elem_len : lenCases) {
              AddNoiseToCdesc(dv, CFI_MAX_RANK);
              check_CFI_establish(
                  dv, base_addr, attribute, type, elem_len, rank, extent);
            }
          }
        }
      }
    }
  }
  // If base_addr is null, extents shall be ignored even if rank !=0
  const int rank3d{3};
  static CFI_CDESC_T(rank3d) dv3darrayStorage;
  CFI_cdesc_t *dv_3darray{reinterpret_cast<CFI_cdesc_t *>(&dv3darrayStorage)};
  AddNoiseToCdesc(dv_3darray, rank3d);  // => dv_3darray->dim[2].extent = -42
  check_CFI_establish(dv_3darray, nullptr, CFI_attribute_other, CFI_type_int, 4,
      rank3d, extents);
  MATCH(false,
      dv_3darray->dim[2].extent == 2 + 66);  // extents was read
}

static void check_CFI_address(
    const CFI_cdesc_t *dv, const CFI_index_t subscripts[]) {
  // 18.5.5.2
  void *addr = CFI_address(dv, subscripts);
  const Descriptor *desc = reinterpret_cast<const Descriptor *>(dv);
  void *addrCheck = desc->Element<void>(subscripts);
  MATCH(true, addr == addrCheck);
}

// Helper function to set lower bound of decsriptor
static void EstablishLowerBounds(CFI_cdesc_t *dv, CFI_index_t *sub) {
  for (int i{0}; i < dv->rank; ++i) {
    dv->dim[i].lower_bound = sub[i];
  }
}

// Helper to get size without making internal compiler functions accessible
static std::size_t ByteSize(CFI_type_t ty, std::size_t size) {
  CFI_CDESC_T(0) storage;
  CFI_cdesc_t *dv = reinterpret_cast<CFI_cdesc_t *>(&storage);
  int retCode =
      CFI_establish(dv, nullptr, CFI_attribute_other, ty, size, 0, nullptr);
  return (retCode == CFI_SUCCESS) ? dv->elem_len : 0;
}

static void run_CFI_address_tests() {
  // Test CFI_address defined in 18.5.5.2
  // Create test world
  CFI_index_t extents[CFI_MAX_RANK];
  CFI_CDESC_T(CFI_MAX_RANK) dv_storage;
  CFI_cdesc_t *dv{reinterpret_cast<CFI_cdesc_t *>(&dv_storage)};
  static char base;
  void *dummyAddr = reinterpret_cast<void *>(&base);
  CFI_attribute_t attrCases[]{
      CFI_attribute_pointer, CFI_attribute_allocatable, CFI_attribute_other};
  CFI_type_t validTypeCases[]{
      CFI_type_int, CFI_type_struct, CFI_type_double, CFI_type_char};
  CFI_index_t subscripts[CFI_MAX_RANK];
  CFI_index_t negativeLowerBounds[CFI_MAX_RANK];
  CFI_index_t zeroLowerBounds[CFI_MAX_RANK];
  CFI_index_t positiveLowerBounds[CFI_MAX_RANK];
  CFI_index_t *lowerBoundCases[] = {
      negativeLowerBounds, zeroLowerBounds, positiveLowerBounds};
  for (int i{0}; i < CFI_MAX_RANK; ++i) {
    negativeLowerBounds[i] = -1;
    zeroLowerBounds[i] = 0;
    positiveLowerBounds[i] = 1;
    extents[i] = i + 2;
    subscripts[i] = i + 1;
  }

  // test for scalar
  for (CFI_attribute_t attribute : attrCases) {
    for (CFI_type_t type : validTypeCases) {
      CFI_establish(dv, dummyAddr, attribute, type, 42, 0, nullptr);
      check_CFI_address(dv, nullptr);
    }
  }
  // test for arrays
  CFI_establish(dv, dummyAddr, CFI_attribute_other, CFI_type_int, 0,
      CFI_MAX_RANK, extents);
  for (CFI_index_t *lowerBounds : lowerBoundCases) {
    EstablishLowerBounds(dv, lowerBounds);
    for (CFI_type_t type : validTypeCases) {
      for (bool contiguous : {true, false}) {
        std::size_t size = ByteSize(type, 12);
        dv->elem_len = size;
        for (int i{0}; i < dv->rank; ++i) {
          // contiguous
          dv->dim[i].sm = size;
          // non contiguous
          dv->dim[i].sm = size + (contiguous ? 0 : dv->elem_len);
          size = dv->dim[i].sm * dv->dim[i].extent;
        }
        for (CFI_attribute_t attribute : attrCases) {
          dv->attribute = attribute;
          check_CFI_address(dv, subscripts);
        }
      }
    }
  }
  // Test on an assumed size array.
  CFI_establish(
      dv, dummyAddr, CFI_attribute_other, CFI_type_int, 0, 3, extents);
  dv->dim[2].extent = -1;
  check_CFI_address(dv, subscripts);
}

static void check_CFI_allocate(CFI_cdesc_t *dv,
    const CFI_index_t lower_bounds[], const CFI_index_t upper_bounds[],
    std::size_t elem_len) {
  // 18.5.5.3
  // Backup descriptor data for futur checks
  const CFI_rank_t rank = dv->rank;
  const std::size_t desc_elem_len = dv->elem_len;
  const CFI_attribute_t attribute = dv->attribute;
  const CFI_type_t type = dv->type;
  const void *base_addr = dv->base_addr;
  const int version = dv->version;
#ifdef VERBOSE
  DumpTestWorld(base_addr, attribute, type, elem_len, rank, nullptr);
#endif
  int retCode = CFI_allocate(dv, lower_bounds, upper_bounds, elem_len);
  Descriptor *desc = reinterpret_cast<Descriptor *>(dv);
  if (retCode == CFI_SUCCESS) {
    // check res properties from 18.5.5.3 par 3
    MATCH(true, dv->base_addr != nullptr);
    for (int i{0}; i < rank; ++i) {
      MATCH(lower_bounds[i], dv->dim[i].lower_bound);
      MATCH(upper_bounds[i], dv->dim[i].extent + dv->dim[i].lower_bound - 1);
    }
    if (type == CFI_type_char) {
      MATCH(elem_len, dv->elem_len);
    } else {
      MATCH(true, desc_elem_len == dv->elem_len);
    }
    MATCH(true, desc->IsContiguous());
  } else {
    MATCH(true, base_addr == dv->base_addr);
  }

  // Below dv members shall not be altered by CFI_allocate regardless of
  // success/failure
  MATCH(true, attribute == dv->attribute);
  MATCH(true, rank == dv->rank);
  MATCH(true, type == dv->type);
  MATCH(true, version == dv->version);

  // Success/failure according to standard
  int numErr{0};
  int expectedRetCode{CFI_SUCCESS};
  if (rank < 0 || rank > CFI_MAX_RANK) {
    ++numErr;
    expectedRetCode = CFI_INVALID_RANK;
  }
  if (type < 0 || type > CFI_type_struct) {
    ++numErr;
    expectedRetCode = CFI_INVALID_TYPE;
  }
  if (base_addr != nullptr && attribute == CFI_attribute_allocatable) {
    // This is less restrictive than 18.5.5.3 arg req for which pointers arg
    // shall be unsassociated. However, this match ALLOCATE behavior
    // (9.7.3/9.7.4)
    ++numErr;
    expectedRetCode = CFI_ERROR_BASE_ADDR_NOT_NULL;
  }
  if (attribute != CFI_attribute_pointer &&
      attribute != CFI_attribute_allocatable) {
    ++numErr;
    expectedRetCode = CFI_INVALID_ATTRIBUTE;
  }
  if (rank > 0 && (lower_bounds == nullptr || upper_bounds == nullptr)) {
    ++numErr;
    expectedRetCode = CFI_INVALID_EXTENT;
  }

  // Memory allocation failures are unpredicatble in this test.
  if (numErr == 0 && retCode != CFI_SUCCESS) {
    MATCH(true, retCode == CFI_ERROR_MEM_ALLOCATION);
  } else if (numErr > 1) {
    MATCH(true, retCode != CFI_SUCCESS);
  } else {
    MATCH(expectedRetCode, retCode);
  }
  // clean-up
  if (retCode == CFI_SUCCESS) {
    CFI_deallocate(dv);
  }
}

static void run_CFI_allocate_tests() {
  // 18.5.5.3
  // create test world
  CFI_CDESC_T(CFI_MAX_RANK) dv_storage;
  CFI_cdesc_t *dv{reinterpret_cast<CFI_cdesc_t *>(&dv_storage)};
  static char base;
  void *dummyAddr = reinterpret_cast<void *>(&base);
  CFI_attribute_t attrCases[]{
      CFI_attribute_pointer, CFI_attribute_allocatable, CFI_attribute_other};
  CFI_type_t typeCases[]{CFI_type_int, CFI_type_struct, CFI_type_double,
      CFI_type_char, CFI_type_other, CFI_type_struct + 1};
  void *baseAddrCases[]{dummyAddr, static_cast<void *>(nullptr)};
  CFI_rank_t rankCases[]{0, 1, CFI_MAX_RANK, CFI_MAX_RANK + 1};
  std::size_t lenCases[]{0, 42};
  CFI_index_t lb1[CFI_MAX_RANK];
  CFI_index_t ub1[CFI_MAX_RANK];
  for (int i{0}; i < CFI_MAX_RANK; ++i) {
    lb1[i] = -1;
    ub1[i] = 0;
  }

  check_CFI_establish(
      dv, nullptr, CFI_attribute_other, CFI_type_int, 0, 0, nullptr);
  for (CFI_type_t type : typeCases) {
    std::size_t ty_len = ByteSize(type, 12);
    for (CFI_attribute_t attribute : attrCases) {
      for (void *base_addr : baseAddrCases) {
        for (CFI_rank_t rank : rankCases) {
          for (size_t elem_len : lenCases) {
            dv->base_addr = base_addr;
            dv->rank = rank;
            dv->attribute = attribute;
            dv->type = type;
            dv->elem_len = ty_len;
            check_CFI_allocate(dv, lb1, ub1, elem_len);
          }
        }
      }
    }
  }
}

static void run_CFI_section_tests() {
  // simple tests
  bool testPreConditions{true};
  constexpr CFI_index_t m = 5, n = 6, o = 7;
  constexpr CFI_rank_t rank = 3;
  long long array[o][n][m];  // Fortran A(m,n,o)
  long long counter{1};

  for (CFI_index_t k{0}; k < o; ++k) {
    for (CFI_index_t j{0}; j < n; ++j) {
      for (CFI_index_t i{0}; i < m; ++i) {
        array[k][j][i] = counter++;  // Fortran A(i,j,k)
      }
    }
  }
  CFI_CDESC_T(rank) sourceStorage;
  CFI_cdesc_t *source = reinterpret_cast<CFI_cdesc_t *>(&sourceStorage);
  CFI_index_t extent[rank] = {m, n, o};
  int retCode = CFI_establish(
      source, &array, CFI_attribute_other, CFI_type_long_long, 0, rank, extent);
  testPreConditions &= (retCode == CFI_SUCCESS);

  CFI_index_t lb[rank] = {2, 5, 4};
  CFI_index_t ub[rank] = {4, 5, 6};
  CFI_index_t strides[rank] = {2, 0, 2};
  constexpr CFI_rank_t resultRank = rank - 1;

  CFI_CDESC_T(resultRank) resultStorage;
  CFI_cdesc_t *result = reinterpret_cast<CFI_cdesc_t *>(&resultStorage);
  retCode = CFI_establish(result, nullptr, CFI_attribute_other,
      CFI_type_long_long, 0, resultRank, nullptr);
  testPreConditions &= (retCode == CFI_SUCCESS);

  if (!testPreConditions) {
    MATCH(true, testPreConditions);
    return;
  }

  retCode = CFI_section(
      result, source, lb, ub, strides);  // Fortran B = A(2:4:2, 5:5:0, 4:6:2)
  MATCH(true, retCode == CFI_SUCCESS);

  const int lbs0 = source->dim[0].lower_bound;
  const int lbs1 = source->dim[1].lower_bound;
  const int lbs2 = source->dim[2].lower_bound;

  CFI_index_t resJ{result->dim[1].lower_bound};
  for (CFI_index_t k{lb[2]}; k <= ub[2]; k += strides[2]) {
    for (CFI_index_t j{lb[1]}; j <= ub[1]; j += (strides[1] ? strides[1] : 1)) {
      CFI_index_t resI{result->dim[0].lower_bound};
      for (CFI_index_t i{lb[0]}; i <= ub[0]; i += strides[0]) {
        // check A(i,j,k) == B(resI, resJ) == array[k-1][j-1][i-1]
        const CFI_index_t resSubcripts[] = {resI, resJ};
        const CFI_index_t srcSubcripts[] = {i, j, k};
        MATCH(true,
            CFI_address(source, srcSubcripts) ==
                CFI_address(result, resSubcripts));
        MATCH(true,
            CFI_address(source, srcSubcripts) ==
                &array[k - lbs2][j - lbs1][i - lbs0]);
        ++resI;
      }
    }
    ++resJ;
  }

  strides[0] = -1;
  lb[0] = 4;
  ub[0] = 2;
  retCode = CFI_section(
      result, source, lb, ub, strides);  // Fortran B = A(4:2:-1, 5:5:0, 4:6:2)
  MATCH(true, retCode == CFI_SUCCESS);

  resJ = result->dim[1].lower_bound;
  for (CFI_index_t k{lb[2]}; k <= ub[2]; k += strides[2]) {
    for (CFI_index_t j{lb[1]}; j <= ub[1]; j += 1) {
      CFI_index_t resI{result->dim[1].lower_bound + result->dim[0].extent - 1};
      for (CFI_index_t i{2}; i <= 4; ++i) {
        // check A(i,j,k) == B(resI, resJ) == array[k-1][j-1][i-1]
        const CFI_index_t resSubcripts[] = {resI, resJ};
        const CFI_index_t srcSubcripts[] = {i, j, k};
        MATCH(true,
            CFI_address(source, srcSubcripts) ==
                CFI_address(result, resSubcripts));
        MATCH(true,
            CFI_address(source, srcSubcripts) ==
                &array[k - lbs2][j - lbs1][i - lbs0]);
        --resI;
      }
    }
    ++resJ;
  }
}

static void run_CFI_select_part_tests() {
  constexpr std::size_t name_len = 5;
  typedef struct {
    double distance;
    int stars;
    char name[name_len];
  } Galaxy;

  const CFI_rank_t rank{2};
  constexpr CFI_index_t universeSize[] = {2, 3};
  Galaxy universe[universeSize[1]][universeSize[0]];

  for (int j{0}; j < universeSize[1]; ++j) {
    for (int i{0}; i < universeSize[0]; ++i) {
      universe[j][i].distance = i + j * 32;
      universe[j][i].stars = i * 2 + j * 64;
      universe[j][i].name[2] = static_cast<char>(i);
      universe[j][i].name[3] = static_cast<char>(j);
    }
  }

  CFI_CDESC_T(rank) resStorage, srcStorage;
  CFI_cdesc_t *result = reinterpret_cast<CFI_cdesc_t *>(&resStorage);
  CFI_cdesc_t *source = reinterpret_cast<CFI_cdesc_t *>(&srcStorage);

  bool testPreConditions{true};
  int retCode = CFI_establish(result, nullptr, CFI_attribute_other,
      CFI_type_int, sizeof(int), rank, nullptr);
  testPreConditions &= (retCode == CFI_SUCCESS);
  retCode = CFI_establish(source, &universe, CFI_attribute_other,
      CFI_type_struct, sizeof(Galaxy), rank, universeSize);
  testPreConditions &= (retCode == CFI_SUCCESS);
  if (!testPreConditions) {
    MATCH(true, testPreConditions);
    return;
  }

  std::size_t displacement{offsetof(Galaxy, stars)};
  std::size_t elem_len{0};  // ignored
  retCode = CFI_select_part(result, source, displacement, elem_len);
  MATCH(CFI_SUCCESS, retCode);

  bool baseAddrShiftedOk =
      (reinterpret_cast<char *>(source->base_addr) + displacement ==
          result->base_addr);
  MATCH(true, baseAddrShiftedOk);
  if (!baseAddrShiftedOk) {
    return;
  }

  MATCH(sizeof(int), result->elem_len);
  for (CFI_index_t j{0}; j < universeSize[1]; ++j) {
    for (CFI_index_t i{0}; i < universeSize[0]; ++i) {
      CFI_index_t subscripts[]{
          result->dim[0].lower_bound + i, result->dim[1].lower_bound + j};
      MATCH(i * 2 + j * 64,
          *reinterpret_cast<int *>(CFI_address(result, subscripts)));
    }
  }

  // Test for Fortran character type
  retCode = CFI_establish(
      result, nullptr, CFI_attribute_other, CFI_type_char, 2, rank, nullptr);
  testPreConditions &= (retCode == CFI_SUCCESS);
  if (!testPreConditions) {
    MATCH(true, testPreConditions);
    return;
  }

  displacement = offsetof(Galaxy, name) + 2;
  elem_len = 2;  // not ignored this time
  retCode = CFI_select_part(result, source, displacement, elem_len);
  MATCH(CFI_SUCCESS, retCode);

  baseAddrShiftedOk =
      (reinterpret_cast<char *>(source->base_addr) + displacement ==
          result->base_addr);
  MATCH(true, baseAddrShiftedOk);
  if (!baseAddrShiftedOk) {
    return;
  }

  MATCH(elem_len, result->elem_len);
  for (CFI_index_t j{0}; j < universeSize[1]; ++j) {
    for (CFI_index_t i{0}; i < universeSize[0]; ++i) {
      CFI_index_t subscripts[]{
          result->dim[0].lower_bound + i, result->dim[1].lower_bound + j};
      MATCH(static_cast<char>(i),
          reinterpret_cast<char *>(CFI_address(result, subscripts))[0]);
      MATCH(static_cast<char>(j),
          reinterpret_cast<char *>(CFI_address(result, subscripts))[1]);
    }
  }
}

static void run_CFI_setpointer_tests() {
  constexpr CFI_rank_t rank = 3;
  CFI_CDESC_T(rank) resStorage, srcStorage;
  CFI_cdesc_t *result = reinterpret_cast<CFI_cdesc_t *>(&resStorage);
  CFI_cdesc_t *source = reinterpret_cast<CFI_cdesc_t *>(&srcStorage);
  CFI_index_t lower_bounds[rank];
  CFI_index_t extents[rank];
  for (int i{0}; i < rank; ++i) {
    lower_bounds[i] = i;
    extents[i] = 2;
  }

  static char target;
  char *dummyBaseAddress = &target;
  bool testPreConditions{true};
  CFI_type_t type{CFI_type_int};
  std::size_t elem_len{ByteSize(type, 42)};
  int retCode = CFI_establish(
      result, nullptr, CFI_attribute_pointer, type, elem_len, rank, nullptr);
  testPreConditions &= (retCode == CFI_SUCCESS);
  retCode = CFI_establish(source, dummyBaseAddress, CFI_attribute_other, type,
      elem_len, rank, extents);
  testPreConditions &= (retCode == CFI_SUCCESS);
  if (!testPreConditions) {
    MATCH(true, testPreConditions);
    return;
  }

  retCode = CFI_setpointer(result, source, lower_bounds);
  MATCH(CFI_SUCCESS, retCode);

  // The follwoing members must be invariant
  MATCH(rank, result->rank);
  MATCH(elem_len, result->elem_len);
  MATCH(type, result->type);
  // check pointer association
  MATCH(true, result->base_addr == source->base_addr);
  for (int j{0}; j < rank; ++j) {
    MATCH(source->dim[j].extent, result->dim[j].extent);
    MATCH(source->dim[j].sm, result->dim[j].sm);
    MATCH(lower_bounds[j], result->dim[j].lower_bound);
  }
}

int main() {
  TestCdescMacroForAllRanksSmallerThan<CFI_MAX_RANK>();
  run_CFI_establish_tests();
  run_CFI_address_tests();
  // TODO: calrify CFI_allocate -> CFI_type_cptr/CFI_type_char
  run_CFI_allocate_tests();
  // TODO: test CFI_deallocate
  // TODO: test CFI_is_contiguous
  run_CFI_section_tests();
  run_CFI_select_part_tests();
  run_CFI_setpointer_tests();
  return testing::Complete();
}
