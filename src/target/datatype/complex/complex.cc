/*!
 * \brief Complex64 Datatype with the Bring Your Own Datatypes (BYODT) framework.
 */
#include <tvm/runtime/c_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>

typedef struct complex64_t {
  float r;
  float i;
} complex64_t;

// Custom datatypes are stored as bits in a uint of the appropriate bit length.
// Thus, when TVM calls these C functions,
// the arguments of are uints that need to reinterpreted as your custom datatype.
//
// When returning, your custom datatype needs to be re-wrapped into a uint,
// which can be thought of as just a wrapper for the raw bits that represent your custom datatype.
TVM_DLL complex64_t Uint64ToComplex64(uint64_t in) {
  // This is a helper function to interpret the uint as your custom dataype.
  complex64_t* t = reinterpret_cast<complex64_t*>(&in); // it is actually needed to define t, otherwise we will get compiler warnings. This line cannot ne combined with the return on the next line
  return *t;
}

TVM_DLL uint64_t Complex64ToUint64(complex64_t in) {
  // This is a helper function to wrap your custom datatype in a uint.
  uint64_t* t = reinterpret_cast<uint64_t*>(&in); // it is actually needed to define t, otherwise we will get compiler warnings. This line cannot ne combined with the return on the next line
  return *t;
}

extern "C" {
TVM_DLL uint64_t MinComplex64() {
  // return minimum representable value
  complex64_t min = {std::numeric_limits<float>::lowest(),std::numeric_limits<float>::lowest()};
  return Complex64ToUint64(min);
}


TVM_DLL float Complex64ToFloat32(uint64_t in) {
// when we convert a complex64 to a float32 we calculate the NORM of the complex number.
  complex64_t t = Uint64ToComplex64(in);
  return sqrtf32(t.r*t.r + t.i*t.i);
}

/* We do not mis-use the float64 datatype anymore!!
TVM_DLL double Complex64ToFloat64(uint64_t in) {
  // cast from complex64 to float64
  double* t = reinterpret_cast<double*>(&in); // it is actually needed to define t, otherwise we will get compiler warnings. This line cannot ne combined with the return on the next line
  return *t;i
}
*/

TVM_DLL uint64_t Float32ToComplex64(float in) {
  // when we convert a float32 to complex64, we make the real part equal to float, and the imaginary part equal to 0.
  // This is especially useful when we want to multiply a real (float32) number with a complex number:
  // First cast the real to a complex number with this function. 
  // Then multiply the two complex numbers
  complex64_t t={in,0.0};
  return Complex64ToUint64(t);
}

TVM_DLL uint64_t Float64ToComplex64(double in) {
  // when we convert a float64 to complex64, we make the real part equal to the double, and the imaginary part equal to 0.
  // This is especially useful when we want to multiply a real (float64) number with a complex number:
  // First cast the real to a complex number with this function. 
  // Then multiply the two complex numbers
  complex64_t t={(float)in,0.0};
  return Complex64ToUint64(t);
}

TVM_DLL uint64_t Int32ToComplex64(int32_t in) {
  complex64_t t={(float)in,0.0};
  return Complex64ToUint64(t);
}

TVM_DLL uint64_t Int64ToComplex64(int64_t in) {
  complex64_t t={(float)in,0.0};
  return Complex64ToUint64(t);
}

TVM_DLL uint64_t UInt32ToComplex64(uint32_t in) {
  complex64_t t={(float)in,0.0};
  return Complex64ToUint64(t);
}
TVM_DLL uint64_t UInt64ToComplex64(uint64_t in) {
  complex64_t t={(float)in,0.0};
  return Complex64ToUint64(t);
}




/* We do not mis-use the float64 datatype anymore!!
TVM_DLL uint64_t Float64ToComplex64(double in) {
  uint64_t* t=reinterpret_cast<uint64_t*>(&in); // it is actually needed to define t, otherwise we will get compiler warnings. This line cannot ne combined with the return on the next line
  return *t;
}
*/

TVM_DLL uint64_t Complex64Add(uint64_t a, uint64_t b) {
  // add operation
  complex64_t in1=Uint64ToComplex64(a);
  complex64_t in2=Uint64ToComplex64(b);

  complex64_t r = {
    in1.r + in2.r,
    in1.i + in2.i
  };

  return Complex64ToUint64(r);
}

TVM_DLL uint64_t Complex64Sub(uint64_t a, uint64_t b) {
  // subtract operation
  complex64_t in1=Uint64ToComplex64(a);
  complex64_t in2=Uint64ToComplex64(b);

  complex64_t r = {
    in1.r - in2.r,
    in1.i - in2.i
  };

  return Complex64ToUint64(r);
}

TVM_DLL uint64_t Complex64Mul(uint64_t a, uint64_t b) {
  // multiply operation
  complex64_t in1=Uint64ToComplex64(a);
  complex64_t in2=Uint64ToComplex64(b);

  complex64_t r = {
    in1.r * in2.r - in1.i * in2.i,
    in1.r * in2.i + in1.i * in2.r
  };

  return Complex64ToUint64(r);
}


TVM_DLL uint64_t Complex64Div(uint64_t a, uint64_t b) {
  // division operation

  // z1=a+b*i, z2=c+d*i
  // z = z1/z2
  //   = (a+b*i)/(c+d*i)
  //   = (a+b*i)*(c+d*i)' / (c+d*i)*(c+d*i)'
  //   = (a+b*i)*(c-d*i) / (c+d*i)*(c-d*i)
  //   = (a*c + b*d) + (b*c - a*d)*i / (c^2+d^2)

  complex64_t in1=Uint64ToComplex64(a);
  complex64_t in2=Uint64ToComplex64(b);

  float numr = in1.r * in2.r + in1.i * in2.i;
  float numi = in1.i * in2.r - in1.r * in2.i;
  float den = in2.r * in2.r + in2.i * in2.i;
  complex64_t r = {
    numr/den,
    numi/den
  };

  return Complex64ToUint64(r);
}

TVM_DLL uint64_t Complex64Conj(uint64_t a) {
  // conjugate operation

  complex64_t in1=Uint64ToComplex64(a);

  complex64_t r = {in1.r, -in1.i};

  return Complex64ToUint64(r);
}

/*
TVM_DLL uint32_t Custom32Max(uint32_t a, uint32_t b) {
  // max
  float acustom = Uint32ToCustom32<float>(a);
  float bcustom = Uint32ToCustom32<float>(b);
  return Custom32ToUint32<float>(acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t Custom32Sqrt(uint32_t a) {
  // sqrt
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(sqrt(acustom));
}

TVM_DLL uint32_t Custom32Exp(uint32_t a) {
  // exponential
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(exp(acustom));
}

TVM_DLL uint32_t Custom32Log(uint32_t a) {
  // log
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(log(acustom));
}

TVM_DLL uint32_t Custom32Sigmoid(uint32_t a) {
  // sigmoid
  float acustom = Uint32ToCustom32<float>(a);
  float one = 1.0f;
  return Custom32ToUint32<float>(one / (one + exp(-acustom)));
}

TVM_DLL uint32_t Custom32Tanh(uint32_t a) {
  // tanh
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(tanh(acustom));
}
*/
} //extern "C"

