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
func @arr2() -> !fir.array<10x10:f32>
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
func @box4() -> !fir.box<!fir.type<derived3{f:f32}>>

func @oth1() -> !fir.dims<1>
func @oth2() -> !fir.field
func @oth3() -> !fir.tdesc<!fir.type<derived5{f1:f32,f2:f32}>>

func @print_index3(index, index, index)
func @user_i64(i64)
func @user_tdesc(!fir.tdesc<!fir.type<x>>)
func @store_tuple(!fir.type<qq1{f1:i32}>) -> ()

func @get_method_box() -> !fir.box<!fir.type<derived3{f:f32}>>
func @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)

func @nop()

func @get_func() -> (() -> ())

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
  %d1:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<100:f32>>, index) -> (index, index, index)
  fir.call @print_index3(%d1#0, %d1#1, %d1#2) : (index, index, index) -> ()
  %8 = fir.call @it1() : () -> !fir.int<4>
  %9 = fir.box_elesize %6 : (!fir.box<!fir.array<100:f32>>) -> i64
  %10 = fir.box_isalloc %6 : (!fir.box<!fir.array<100:f32>>) -> i1
  %11 = fir.box_isarray %6 : (!fir.box<!fir.array<100:f32>>) -> i1
  %12 = fir.box_isptr %6 : (!fir.box<!fir.array<100:f32>>) -> i1
  %13 = fir.box_rank %6 : (!fir.box<!fir.array<100:f32>>) -> i64
  %14 = fir.box_tdesc %6 : (!fir.box<!fir.array<100:f32>>) -> !fir.tdesc<!fir.array<100:f32>>
  %15 = fir.call @box2() : () -> !fir.boxchar<2>
  %16 = fir.boxchar_len %15 : (!fir.boxchar<2>) -> i32
  %17 = fir.call @box3() : () -> !fir.boxproc<(i32, i32) -> i64>
  %18 = fir.boxproc_host %17 : (!fir.boxproc<(i32, i32) -> i64>) -> !fir.ref<i32>
  %19 = constant 10 : i32
  %20 = fir.coordinate_of %5, %19 : (!fir.heap<!fir.array<100:f32>>, i32) -> !fir.ref<f32>
  %21 = fir.field_index("f") : !fir.field
  %22 = fir.undefined !fir.type<derived{f:f32}>
  %23 = fir.extract_value %22, %21 : (!fir.type<derived{f:f32}>, !fir.field) -> f32
  %c1 = constant 1 : i32
  %24 = fir.gendims %c1, %19, %c1 : (i32, i32, i32) -> !fir.dims<1>
  %cf1 = constant 1.0 : f32
  %25 = fir.insert_value %22, %cf1, %21 : (!fir.type<derived{f:f32}>, f32, !fir.field) -> !fir.type<derived{f:f32}>
  %26 = fir.len_param_index("f") : !fir.field
  %27 = fir.call @box4() : () -> !fir.box<!fir.type<derived3{f:f32}>>
  %28 = fir.dispatch "method"(%27) : (!fir.box<!fir.type<derived3{f:f32}>>) -> i32
  %29 = fir.convert %28 : (i32) -> i64
  %30 = fir.gentypedesc !fir.type<x> : !fir.tdesc<!fir.type<x>>
  fir.call @user_tdesc(%30) : (!fir.tdesc<!fir.type<x>>) -> ()
  %31 = fir.no_reassoc %29 : i64
  fir.call @user_i64(%31) : (i64) -> ()
  fir.freemem %5 : !fir.heap<!fir.array<100:f32>>
  %32 = fir.call @get_func() : () -> (() -> ())
  fir.call %32() : () -> ()
  %33 = fir.address_of (@it1) : !fir.ref<() -> !fir.int<4>>
  return
}

func @boxing_match() {
  %0 = fir.alloca i32 : !fir.ref<i32>
  %1 = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
  %2:6 = fir.unbox %1 : (!fir.box<i32>) -> (!fir.ref<i32>,i32,i32,!fir.tdesc<i32>,i32,!fir.dims<0>)
  %c8 = constant 8 : i32
  %3 = fir.alloca !fir.char<1> : !fir.ref<!fir.char<1>>
  %b3 = fir.undefined !fir.char<1>
  %4 = fir.emboxchar %3, %c8 : (!fir.ref<!fir.char<1>>, i32) -> !fir.boxchar<1>
  %5:2 = fir.unboxchar %4 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i32)
  %6 = fir.alloca !fir.type<qq2{f1:i32,f2:f64}> : !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>
  %a1 = fir.undefined !fir.type<qq2{f1:i32,f2:f64}>
  %z = constant 0 : i32
  %c12 = constant 12 : i32
  %a2 = fir.insert_value %a1, %c12, %z : (!fir.type<qq2{f1:i32,f2:f64}>, i32, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
  %z1 = constant 1 : i32
  %c42 = constant 42.13 : f64
  %a3 = fir.insert_value %a1, %c42, %z1 : (!fir.type<qq2{f1:i32,f2:f64}>, f64, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
  fir.store %a3 to %6 : !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>
  %7 = fir.emboxproc @method_impl, %6 : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
  %8:2 = fir.unboxproc %7 : (!fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>) -> ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>)
  %9 = call @box3() : () -> !fir.boxproc<(i32, i32) -> i64>
  %10:2 = fir.unboxproc %9 : (!fir.boxproc<(i32, i32) -> i64>) -> ((i32, i32) -> i64, !fir.ref<!fir.type<qq1{f1:i32}>>)
  %11 = fir.load %10#1 : !fir.ref<!fir.type<qq1{f1:i32}>>
  fir.call @store_tuple(%11) : (!fir.type<qq1{f1:i32}>) -> ()
  return
}

func @loop() {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %ct = constant true
  fir.loop %i = %c1 to %c10 {
    fir.where %ct {
      fir.call @nop() : () -> ()
    } otherwise {
      fir.call @nop() : () -> ()
    }
  }
  fir.unreachable
}

func @bar_select(%arg : i32, %arg2 : i32) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  fir.select %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), -3,^bb3(%arg2,%2:i32,i32), 4,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2 : i32, %b3:i32) :
  %4 = addi %b, %b2 : i32
  %5 = addi %4, %b3 : i32
  return %5 : i32
^bb3(%c:i32, %c2:i32) :
  %6 = addi %c, %c2 : i32
  return %6 : i32
^bb4(%d : i32) :
  return %d : i32
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

func @bar_select_rank(%arg : i32, %arg2 : i32) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  fir.select_rank %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), 3,^bb3(%arg2,%2:i32,i32), -1,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2 : i32, %b3:i32) :
  %4 = addi %b, %b2 : i32
  %5 = addi %4, %b3 : i32
  return %5 : i32
^bb3(%c:i32, %c2:i32) :
  %6 = addi %c, %c2 : i32
  return %6 : i32
^bb4(%d : i32) :
  return %d : i32
^bb5 :
  %zero = constant 0 : i32
  %7 = fir.call @get_method_box() : () -> !fir.box<!fir.type<derived3{f:f32}>>
  fir.dispatch method(%7) : (!fir.box<!fir.type<derived3{f:f32}>>) -> ()
  return %zero : i32
}

func @bar_select_type(%arg : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>>) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  fir.select_type %arg : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>> [ #fir.instance<i32>,^bb1(%0:i32), #fir.instance<i64>,^bb2(%2:i32), #fir.subsumed<i16>,^bb3(%2:i32), #fir.instance<i8>,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32) :
  return %b : i32
^bb3(%c : i32) :
  return %c : i32
^bb4(%d : i32) :
  return %d : i32
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

func @bar_select_case(%arg : i32, %arg2 : i32) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  fir.select_case %arg : i32 [#fir.point, %0, ^bb1(%0:i32), #fir.lower, %1, ^bb2(%2,%arg,%arg2,%1:i32,i32,i32,i32), #fir.interval, %2, %3, ^bb3(%2,%arg2:i32,i32), #fir.upper, %arg, ^bb4(%1:i32), unit, ^bb5]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2:i32, %b3:i32, %b4:i32) :
  %4 = addi %b, %b2 : i32
  %5 = muli %4, %b3 : i32
  %6 = addi %5, %b4 : i32
  return %6 : i32
^bb3(%c : i32, %c2 : i32) :
  %7 = addi %c, %c2 : i32
  return %7 : i32
^bb4(%d : i32) :
  return %d : i32
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

fir.global @global_var : i32 {
  %0 = constant 1 : i32
}

fir.global @global_constant constant : i32 {
  %0 = constant 934 : i32
}

fir.global @global_derived : !fir.type<mine{f:i32}> {
  %0 = constant 1 : i32
  fir.global_entry "f", %0 : i32
}

fir.dispatch_table @dispatch_tbl {
  fir.dt_entry "method", @method_impl
}
