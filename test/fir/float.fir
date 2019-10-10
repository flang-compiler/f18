func @bar(%a : !fir.real<2>, %b : !fir.real<4>, %c : !fir.real<8>, %d : !fir.real<10>, %e : !fir.real<16>) -> !fir.real<10> {
  %1 = fir.convert %a : (!fir.real<2>) -> !fir.real<10>
  %2 = fir.convert %b : (!fir.real<4>) -> !fir.real<10>
  %3 = fir.convert %c : (!fir.real<8>) -> !fir.real<10>
  %4 = fir.convert %d : (!fir.real<10>) -> !fir.real<10>
  %5 = fir.convert %e : (!fir.real<16>) -> !fir.real<10>
  %6 = call @foo2(%1, %2, %3, %4, %5) : (!fir.real<10>, !fir.real<10>, !fir.real<10>, !fir.real<10>, !fir.real<10>) -> !fir.real<10>
  return %6 : !fir.real<10>
}

func @foo(%a : !fir.real<16>, %b : !fir.real<16>, %c : !fir.real<16>, %d : !fir.real<16>, %e : !fir.real<16>) -> !fir.real<16> {
  %1 = fir.addf %a, %b : !fir.real<16>
  %2 = fir.mulf %1, %c : !fir.real<16>
  %3 = fir.subf %2, %d : !fir.real<16>
  %4 = fir.divf %3, %e : !fir.real<16>
  %5 = fir.modf %4, %a : !fir.real<16>
  return %5 : !fir.real<16>
}

func @foo2(%a : !fir.real<10>, %b : !fir.real<10>, %c : !fir.real<10>, %d : !fir.real<10>, %e : !fir.real<10>) -> !fir.real<10> {
  %1 = fir.addf %a, %b : !fir.real<10>
  %2 = fir.mulf %1, %c : !fir.real<10>
  %3 = fir.subf %2, %d : !fir.real<10>
  %4 = fir.divf %3, %e : !fir.real<10>
  %5 = fir.modf %4, %a : !fir.real<10>
  return %5 : !fir.real<10>
}
