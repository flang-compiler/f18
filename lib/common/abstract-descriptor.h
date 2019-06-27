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

#ifndef FORTRAN_COMMON_ABSTRACT_DESCRIPTOR_H_
#define FORTRAN_COMMON_ABSTRACT_DESCRIPTOR_H_

#include "constexpr-bitset.h"

namespace Fortran::common {

// DescriptorInterface both serves to describe and validate the interface
// that descriptor implementations must comply with in order to be used to
// instantiate Fortran transformational intrinsic abstract implementation.
// To validate the compliance of a descriptor implementation DESC with the
// interface, one must compile in a test file:
//
//  template class DescriptorInterface<DESC>;
//  template class Transformational<DescriptorInterface<DESC>>;
//
// The first line ensures DESC has all properties required by the descriptor
// interface. The second line ensures the Abstract Fortran Runtime
// implementation is only using descriptor properties that are part of the
// descriptor interface. These are static tests only and it does not validate
// the descriptor implementation logic.
// Note that inheritance and abstract functions were not selected to make the
// interface to avoid runtime overhead with virtual functions.

template<typename DESC> class DescriptorInterface : private DESC {
public:
  // A descriptor implementation must define the following type

  // basic integer types
  using SubscriptValue = typename DESC::SubscriptValue;
  using SizeValue = typename DESC::SizeValue;  // may or may not be unsigned
  using Attribute = typename DESC::Attribute;

  // RankedSizedArray<A>(n) are arrays with storage for at least n A elements.
  // The implementation is free to allocate more memory space than needed.
  // n must not be greater than maxRank;
  template<typename A>
  using RankedSizedArray = typename DESC::template RankedSizedArray<A>;

  // SubscriptArray is an alias for RankedSizedArray<SubscriptValue>
  using SubscriptArray = typename DESC::SubscriptArray;

  // OwningPointer<A> acts like a A*. When receiving an OwningPointer<A>,
  // one is responsible to destroy its content. It may be implemented as
  // a simple A* or something more complex and safer.
  template<typename A>
  using OwningPointer = typename DESC::template OwningPointer<A>;

  // TODO: Describe FortranType and Dimension interface or hide them
  // FortranType holds type information
  using FortranType = typename DESC::FortranType;
  // Dimension holds dimension information.
  using Dimension = typename DESC::Dimension;

  // Constants to be defined by descriptor implementation
  static constexpr int maxRank{DESC::maxRank};

  // A descriptor implementation must provide the following functions:

  int rank() const { return DESC::rank(); }

  FortranType type() const { return DESC::type(); }

  Dimension GetDimension(int i) const { return DESC::GetDimension(i); }

  // Write lower bounds of the entity describe by the descriptor
  // into subscripts.
  void GetLowerBounds(SubscriptArray &subscripts) const {
    DESC::GetLowerBounds(subscripts);
  }

  bool HasSameTypeAs(const DescriptorInterface<DESC> &other) const {
    return DESC::HasSameTypeAs(static_cast<const DESC &>(other));
  }

  // When the decsriptor describes an array containing integer,
  // GetSubscriptValueAt extract the element indexed by subscripts and convert
  // it to a SubscriptValue.
  SubscriptValue GetSubscriptValueAt(const SubscriptArray &subscripts) const {
    return DESC::GetSubscriptValueAt(subscripts);
  }

  // Number of elements
  SizeValue Elements() const { return DESC::Elements(); }

  // Create a descriptor with the same type (including type parameters if
  // applicable) as source and the given rank. The data and bounds are left
  // unset.
  static OwningPointer<DescriptorInterface<DESC>> CreateWithSameTypeAs(
      const DescriptorInterface<DESC> &source, int rank, Attribute attribute) {
    return OwningPointer<DescriptorInterface<DESC>>{
        static_cast<DescriptorInterface<DESC> *>(&(*DESC::CreateWithSameTypeAs(
            static_cast<const DESC &>(source), rank, attribute)))};
  }

  // Allocate a descriptor for which the type is fully defined:
  //  - set bounds information (the lower bounds are set to to 1)
  //  - get storage for the entity described by the descriptor and
  //    make it available to the descriptor.
  //  The data is not initialized (even for derived type).
  void Allocate(const SubscriptArray &extents) { DESC::Allocate(extents); }

  // Copy count elements from the entity described by source into the entity
  // describe by the descriptor.
  // - source and the descriptor must describe entities of the same type. source
  // must have at least one element.
  // - For the target, the copy starts at the element indexed by resultSubscript
  // and resultSubscript are incremented according to the dimension order in
  // dimOrder.
  // - For the source, the copy starts at the first element and source elements
  // are run through in Fortran dimension order. If the end of the source of the
  // target is reached before count, the copy continues from the first element
  // of the entity whose last element was reached.
  SizeValue CopyFrom(const DescriptorInterface<DESC> &source,
      SubscriptArray &resultSubscripts, SizeValue count,
      const RankedSizedArray<int> *dimOrder) {
    return DESC::CopyFrom(
        static_cast<const DESC &>(source), resultSubscripts, count, dimOrder);
  }

  // Indicate that the descriptor is not describing a user variable (this
  // information may not be useful and this may be a no-op)
  void SetCompilerCreatedFlag() { DESC::SetCompilerCreatedFlag(); }

  // TODO: improve interface for attributes
  static Attribute AllocatableAttribute() {
    return DESC::AllocatableAttribute();
  }
};

namespace safestd {
// This namespace contains symbols named after C++ standard library that
// have the same interface as the corresponding STL symbols.
// It is guaranteed that there usage will not bring
// dependencies with the C++ runtime library.
// The purpose is to avoid having the Fortran runtime be dependent upon the
// C++ runtime while allowing Fortran runtime programmers to use C++ features
// that they are used to.
// A symbols can directly be an alias of the corresponding std symbol when its
// usage is not dependent on the C++ runtime. These independence is determined
// empirically by testing that the Fortran runtime can be built without linking
// to the C++ runtime. This has to be shown with all the supported f18 built
// toolchain.
// If a C++ standard library feature depends on C++ runtime and is highly
// desired, a C++ runtime independent implementation will need to be provided
// here. Similarly, if a new toolchain introduces C++ runtime dependencies for
// the symbols previously listed here, re-implementation of these features will
// have to be done as part of the toolchain support introduction.

// Alas, std::bitset comes with C++ exceptions runtime symbols (even with
// -fno-exceptions).
template<std::size_t N> using bitset = Fortran::common::BitSet<N>;
template<typename T> constexpr T min(const T &a, const T &b) {
  return (a < b ? a : b);
}
}

// Transformational contains an implementation of Fortran intrinsic functions
// operating on descriptors in a descriptor abstracted way. The purpose is that
// these implementations can be used both in the front-end for folding and the
// runtime library, by selecting the applicable descriptor. DESC has to comply
// with the DescriptorInterface above. Because these functions can be
// instantiated in a runtime library context, their abstract implementations
// must not use any C++ features that would bring runtime dependencies.
template<typename DESC> class Transformational {
  // TODO: abstract error handling and memory allocation
public:
  using SubscriptValue = typename DESC::SubscriptValue;
  using SizeValue = typename DESC::SizeValue;
  template<typename A>
  using RankedSizedArray = typename DESC::template RankedSizedArray<A>;
  using SubscriptArray = typename DESC::SubscriptArray;
  template<typename A>
  using OwningPointer = typename DESC::template OwningPointer<A>;
  using FortranType = typename DESC::FortranType;
  using Dimension = typename DESC::Dimension;
  using Attribute = typename DESC::Attribute;
  static constexpr int maxRank{DESC::maxRank};

  static OwningPointer<DESC> RESHAPE(const DESC &source, const DESC &shape,
      const DESC *pad = nullptr, const DESC *order = nullptr) {
    // Compute and check the rank of the result.
    CHECK(shape.rank() == 1);
    CHECK(shape.type().IsInteger());
    SubscriptValue resultRank{shape.GetDimension(0).Extent()};
    CHECK(
        resultRank >= 0 && resultRank <= static_cast<SubscriptValue>(maxRank));

    // Extract and check the shape of the result; compute its element count.
    SubscriptArray resultExtent(resultRank);
    SizeValue resultElements{1};
    SubscriptArray shapeSubscript(1);
    shapeSubscript[0] = shape.GetDimension(0).LowerBound();
    for (SubscriptValue j{0}; j < resultRank; ++j, shapeSubscript[0]++) {
      resultExtent[j] = shape.GetSubscriptValueAt(shapeSubscript);
      CHECK(resultExtent[j] >= 0);
      resultElements *= resultExtent[j];
    }

    // Check that there are sufficient elements in the SOURCE=, or that
    // the optional PAD= argument is present and nonempty.
    SizeValue sourceElements{source.Elements()};
    SizeValue padElements{pad ? pad->Elements() : 0};
    if (sourceElements < resultElements) {
      CHECK(padElements > 0);
      CHECK(pad->HasSameTypeAs(source));
    }

    // Extract and check the optional ORDER= argument, which must be a
    // permutation of [1..resultRank].
    RankedSizedArray<int> dimOrder(resultRank);
    if (order != nullptr) {
      CHECK(order->rank() == 1);
      CHECK(order->type().IsInteger());
      CHECK(order->GetDimension(0).Extent() == resultRank);
      safestd::bitset<maxRank> values;
      SubscriptArray orderSubscript(1);
      orderSubscript[0] = order->GetDimension(0).LowerBound();
      for (SubscriptValue j{0}; j < resultRank; ++j, ++orderSubscript[0]) {
        auto k{order->GetSubscriptValueAt(orderSubscript)};
        CHECK(k >= 1 && k <= resultRank && !values.test(k - 1));
        values.set(k - 1);
        dimOrder[k - 1] = j;
      }
    } else {
      for (int j{0}; j < resultRank; ++j) {
        dimOrder[j] = j;
      }
    }

    // Create and populate the result's descriptor.
    OwningPointer<DESC> result{DESC::CreateWithSameTypeAs(
        source, resultRank, DESC::AllocatableAttribute())};
    result->SetCompilerCreatedFlag();
    result->Allocate(resultExtent);
    // Copy elements
    SubscriptArray resultSubscript(resultRank);
    result->GetLowerBounds(resultSubscript);
    SizeValue count{safestd::min(sourceElements, resultElements)};
    SizeValue copied{
        result->CopyFrom(source, resultSubscript, count, &dimOrder)};
    if (copied < resultElements) {
      CHECK(pad);
      result->CopyFrom(
          *pad, resultSubscript, resultElements - copied, &dimOrder);
    }
    return result;
  }
};

}

#endif  // FORTRAN_COMMON_ABSTRACT_DESCRIPTOR_H_
