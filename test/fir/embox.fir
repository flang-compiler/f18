#x0 = (d0, d1) -> (d1, d0)

func @f(%arg : !fir.ref<!fir.array<?:f64>>) {
  %1 = fir.embox %arg [#x0] : (!fir.ref<!fir.array<?:f64>>) -> !fir.box<!fir.array<?:f64>, >
  return
}
