func @add(%a : !fir.complex<4>, %b : !fir.complex<4>) -> !fir.complex<4>

func @foo(%a : !fir.complex<4>, %b : !fir.complex<4>) -> !fir.complex<4> {
  %1 = call @add(%a, %b) : (!fir.complex<4>, !fir.complex<4>) -> !fir.complex<4>
  return %1 : !fir.complex<4>
}
