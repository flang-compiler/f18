func @foo(%a : !fir.complex<4>, %b : !fir.complex<4>, %c : !fir.complex<4>, %d : !fir.complex<4>, %e : !fir.complex<4>) -> !fir.complex<4> {
  %1 = fir.addc %a, %b : !fir.complex<4>
  %2 = fir.mulc %1, %c : !fir.complex<4>
  %3 = fir.subc %2, %d : !fir.complex<4>
  %4 = fir.divc %3, %e : !fir.complex<4>
  return %4 : !fir.complex<4>
}
