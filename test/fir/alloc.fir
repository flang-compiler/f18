func @f1() -> !fir.ref<i32> {
  %1 = fir.alloca i32 : !fir.ref<i32>
  return %1 : !fir.ref<i32>
}

func @f2() -> !fir.ref<i32> {
  %0 = constant 100 : index
  %1 = fir.alloca i32, %0 : !fir.ref<i32>
  return %1 : !fir.ref<i32>
}

func @f3() -> !fir.heap<i32> {
  %1 = fir.allocmem i32 : !fir.heap<i32>
  return %1 : !fir.heap<i32>
}

func @f4() -> !fir.heap<i32> {
  %0 = constant 100 : index
  %1 = fir.allocmem i32, %0 : !fir.heap<i32>
  return %1 : !fir.heap<i32>
}
