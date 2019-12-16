func @f(%a : !fir.box<none>) -> i32 {
   %1 = constant 4 : i32
   %2 = constant 8 : i32
   %3 = constant 16 : i32
   fir.select_type %a : !fir.box<none> [#fir.instance<!fir.type<a>>, ^bb2(%1:i32), #fir.subsumed<!fir.type<b>>, ^bb3(%3:i32), unit, ^bb4(%2:i32)]
^bb2(%4 : i32) :
   return %4 : i32
^bb3(%5 : i32) :
   %6 = addi %5, %5 : i32
   return %6 : i32
^bb4(%7 : i32) :
   %8 = muli %7, %7 : i32
   return %8 : i32
}
