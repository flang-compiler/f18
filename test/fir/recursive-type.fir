!t = type !fir.type<t {p : !fir.ptr<!fir.type<t>>}>

func @a(%a : !t) {
  return
}
