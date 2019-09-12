// Test the FIR operations

func @it1() -> !fir.int<4>
func @it2() -> !fir.real<8>
func @it3() -> !fir.complex<8>
func @it4() -> !fir.logical<1>
func @it5() -> !fir.char<1>

func @dvd1() -> !fir.type<derived1>
func @dvd2() -> !fir.type<derived2(p:i32)>
func @dvd3() -> !fir.type<derived3{f:f32}>
func @dvd4() -> !fir.type<derived4(p:i8){f:f32}>

func @arr1() -> !fir.array<10:f32>
//func @arr2() -> !fir.array<10x10:f32>
func @arr3() -> !fir.array<?:f32>
func @arr4() -> !fir.array<10x?:f32>
func @arr5() -> !fir.array<?x?:f32>
func @arr6() -> !fir.array<*:f32>

func @mem1() -> !fir.ref<i32>
func @mem2() -> !fir.ptr<i32>
func @mem3() -> !fir.heap<i32>
func @mem4() -> !fir.ref<() -> ()>

func @box1() -> !fir.box<!fir.array<?:f32>>
func @box2() -> !fir.boxchar<2>
func @box3() -> !fir.boxproc<(i32, i32) -> i64>

func @oth1() -> !fir.dims<1>
func @oth2() -> !fir.field
func @oth3() -> !fir.tdesc<!fir.type<derived5{f1:f32,f2:f32}>>

func @instructions() {
  %0 = fir.alloca !fir.array<10:i32> : !fir.ref<!fir.array<10:i32>>
  %1 = fir.load %0 : !fir.ref<!fir.array<10:i32>>
  %2 = fir.alloca i32 : !fir.ref<i32>
  %3 = constant 22 : i32
  fir.store %3 to %2 : !fir.ref<i32>
  %4 = fir.undefined i32
  %5 = fir.allocmem !fir.array<100:f32> : !fir.heap<!fir.array<100:f32>>
  %6 = fir.embox %5 : (!fir.heap<!fir.array<100:f32>>) -> !fir.box<!fir.array<100:f32>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.array<100:f32>>) -> !fir.ref<!fir.array<100:f32>>
  %c0 = constant 0 : index
//  %8 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<100:f32>>) -> (index, index, index)
//  %8 = fir.call @it1() : () -> !fir.int<4>
  %8 = call @it1() : () -> !fir.int<4>
  %9 = fir.box_elesize %6 : (!fir.box<!fir.array<100:f32>>) -> i64
  %10 = fir.box_isalloc %6 : (!fir.box<!fir.array<100:f32>>) -> i1
  %11 = fir.box_isarray %6 : (!fir.box<!fir.array<100:f32>>) -> i1
  %12 = fir.box_isptr %6 : (!fir.box<!fir.array<100:f32>>) -> i1
  %13 = fir.box_rank %6 : (!fir.box<!fir.array<100:f32>>) -> i64
  %14 = fir.box_tdesc %6 : (!fir.box<!fir.array<100:f32>>) -> !fir.tdesc<!fir.array<100:f32>>
//  %15 = call @box2() : () -> !fir.boxchar<2>
  %15 = call @it1() : () -> !fir.int<4>
//  %16 = fir.boxchar_len %15 : (!fir.boxchar<2>) -> i32
  %16 = call @it1() : () -> !fir.int<4>
  %17 = call @box3() : () -> !fir.boxproc<(i32, i32) -> i64>
//  %18 = fir.boxproc_host %17 : (!fir.boxproc<(i32, i32) -> i64>) -> (!fir.ref<(i32, i32) -> i64>, !fir.ref<(i32, f64)>)
  %18,%19 = fir.boxproc_host %17 : (!fir.boxproc<(i32, i32) -> i64>) -> (!fir.ref<(i32, i32) -> i64>, !fir.ref<i32>)
  %c10 = constant 10 : i32
  %20 = fir.coordinate_of %5, %c10 : (!fir.heap<!fir.array<100:f32>>, i32) -> !fir.ref<f32>
//  %21 = fir.field_index "f" : () -> !fir.field
  %21 = fir.undefined !fir.field
//  %22 = call @dvd4() : () -> !fir.type<derived4(p:i8){f:f32}>
  %22 = fir.undefined !fir.type<derived{f:f32}>
  %23 = fir.extract_value %22, %21 : (!fir.type<derived{f:f32}>, !fir.field) -> f32
  %c1 = constant 1 : i32
  %24 = fir.gendims %c1, %c10, %c1 : (i32, i32, i32) -> !fir.dims<1>
  %cf1 = constant 1.0 : f32
  %25 = fir.insert_value %22, %cf1, %21 : (!fir.type<derived{f:f32}>, f32, !fir.field) -> !fir.type<derived{f:f32}>
//  %26 = fir.len_param_index "f" : (!fir.field) -> i32
  %26 = fir.undefined i32
//  %27 = fir.call @dvd3() : () -> !fir.type<derived3{f:f32}>
  %27 = fir.undefined !fir.type<derived3{f:f32}>
//  %28 = fir.dispatch "method"(%27) : () -> i32
  %28 = fir.undefined i32
  %29 = fir.convert %26 : (i32) -> i64
  %30 = fir.gentypedesc !fir.type<x> : !fir.tdesc<!fir.type<x>>
  %31 = fir.no_reassoc %29 : i64
  fir.freemem %5 : !fir.heap<!fir.array<100:f32>>
  return
}
