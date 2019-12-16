func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  fir.loop %iv = %lb to %ub step %step unordered {
    fir.where %b {
      fir.store %iv to %addr : !fir.ref<index>
    } otherwise {
      %zero = constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}
