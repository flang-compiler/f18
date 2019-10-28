func @f(%a : i32) -> i32 {
   %1 = constant 1 : i32
   %2 = constant 42 : i32
   fir.select %a : i32 [1, ^bb2(%1:i32), unit, ^bb3(%2:i32)]
^bb2(%3 : i32) :
   return %3 : i32
^bb3(%4 : i32) :
   %5 = addi %4, %4 : i32
   return %5 : i32 
}

func @g(%a : i32) -> i32 {
   %1 = constant 1 : i32
   %2 = constant 42 : i32
   fir.select_rank %a : i32 [1, ^bb2(%1:i32), -1, ^bb4, unit, ^bb3(%2:i32)]
^bb2(%3 : i32) :
   return %3 : i32
^bb3(%4 : i32) :
   %5 = addi %4, %4 : i32
   return %5 : i32 
^bb4:
   return %a : i32
}

func @h(%a : i32) -> i32 {
   %1 = constant 1 : i32
   %2 = constant 42 : i32
   %b1 = constant 4 : i32
   %b2 = constant 14 : i32
   %b3 = constant 82 : i32
   %b4 = constant 96 : i32
   fir.select_case %a : i32 [#fir.point, %1, ^bb2(%1:i32), #fir.lower, %b1, ^bb4, #fir.upper, %b2, ^bb6, #fir.interval, %b3, %b4, ^bb5, unit, ^bb3(%2:i32)]
^bb2(%3 : i32) :
   return %3 : i32
^bb3(%4 : i32) :
   %5 = addi %4, %4 : i32
   br ^bb2(%5 : i32) 
^bb4:
   return %a : i32
^bb5:
   return %1 : i32
^bb6:
   %x = addi %b4, %b3 : i32
   return %x : i32
}
