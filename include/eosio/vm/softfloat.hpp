#pragma once
#ifdef EOS_VM_SOFTFLOAT
#include <eosio/vm/exceptions.hpp>

#include <cstdint>

#include <softfloat.hpp>

namespace eosio { namespace vm {
inline constexpr uint32_t inv_float_eps = 0x4B000000;
inline constexpr uint64_t inv_double_eps = 0x4330000000000000;

inline bool is_nan( const float32_t f ) {
   return f32_is_nan( f );
}
inline bool is_nan( const float64_t f ) {
   return f64_is_nan( f );
}

template<bool ForceArithmetic>
float32_t propagate_nan(float32_t a) {
   if constexpr(ForceArithmetic) {
      return { a.v | 0x00400000u };
   } else {
      return a;
   }
}

template<bool ForceArithmetic>
float64_t propagate_nan(float64_t a) {
   if constexpr(ForceArithmetic) {
      return { a.v | 0x0008000000000000u };
   } else {
      return a;
   }
}

inline float _eosio_f32_add( float a, float b ) {
   return ::from_softfloat32( ::f32_add( ::to_softfloat32(a), ::to_softfloat32(b) ) );
}

inline float _eosio_f32_sub( float a, float b ) {
   return ::from_softfloat32( ::f32_sub( ::to_softfloat32(a), ::to_softfloat32(b) ) );
}

inline float _eosio_f32_div( float a, float b ) {
   return ::from_softfloat32( ::f32_div( ::to_softfloat32(a), ::to_softfloat32(b) ) );
}

inline float _eosio_f32_mul( float a, float b ) {
   return ::from_softfloat32( ::f32_mul( ::to_softfloat32(a), ::to_softfloat32(b) ) );
}

template<bool ArithNan>
inline float _eosio_f32_min( float af, float bf ) {
   float32_t a = to_softfloat32(af);
   float32_t b = to_softfloat32(bf);
   if (is_nan(a)) {
      return from_softfloat32(propagate_nan<ArithNan>(a));
   }
   if (is_nan(b)) {
      return from_softfloat32(propagate_nan<ArithNan>(b));
   }
   if ( f32_sign_bit(a) != f32_sign_bit(b) ) {
      return f32_sign_bit(a) ? af : bf;
   }
   return ::f32_lt(a,b) ? af : bf;
}

template<bool ArithNan>
inline float _eosio_f32_max( float af, float bf ) {
   float32_t a = to_softfloat32(af);
   float32_t b = to_softfloat32(bf);
   if (is_nan(a)) {
      return from_softfloat32(propagate_nan<ArithNan>(a));
   }
   if (is_nan(b)) {
      return from_softfloat32(propagate_nan<ArithNan>(b));
   }
   if ( f32_sign_bit(a) != f32_sign_bit(b) ) {
      return f32_sign_bit(a) ? bf : af;
   }
   return ::f32_lt( a, b ) ? bf : af;
}

inline float _eosio_f32_copysign( float af, float bf ) {
   float32_t a = to_softfloat32(af);
   float32_t b = to_softfloat32(bf);
   a.v &= ~(1 << 31);             // clear the sign bit
   a.v = a.v | ((b.v >> 31) << 31); // add the sign of b
   return from_softfloat32(a);
}

inline float _eosio_f32_abs( float af ) {
   float32_t a = to_softfloat32(af);
   a.v &= ~(1 << 31);
   return from_softfloat32(a);
}

inline float _eosio_f32_neg( float af ) {
   float32_t a = to_softfloat32(af);
   uint32_t sign = a.v >> 31;
   a.v &= ~(1 << 31);
   a.v |= (!sign << 31);
   return from_softfloat32(a);
}

inline float _eosio_f32_sqrt( float a ) {
   float32_t ret = ::f32_sqrt( to_softfloat32(a) );
   return from_softfloat32(ret);
}

template<bool ArithNan>
inline float _eosio_f32_ceil( float af ) {
   float32_t a = to_softfloat32(af);
   int e = (int)(a.v >> 23 & 0xFF) - 0X7F;
   uint32_t m;
   if (ArithNan && is_nan(a)) {
      return from_softfloat32(propagate_nan<ArithNan>(a));
   }
   if (e >= 23)
      return af;
   if (e >= 0) {
      m = 0x007FFFFF >> e;
      if ((a.v & m) == 0)
         return af;
      if (a.v >> 31 == 0)
         a.v += m;
      a.v &= ~m;
   } else {
      if (a.v >> 31)
         a.v = 0x80000000; // return -0.0f
      else if (a.v << 1)
         a.v = 0x3F800000; // return 1.0f
   }

   return from_softfloat32(a);
}

template<bool ArithNan>
inline float _eosio_f32_floor( float af ) {
   float32_t a = to_softfloat32(af);
   int e = (int)(a.v >> 23 & 0xFF) - 0X7F;
   uint32_t m;
   if (ArithNan && is_nan(a)) {
      return from_softfloat32(propagate_nan<ArithNan>(a));
   }
   if (e >= 23)
      return af;
   if (e >= 0) {
      m = 0x007FFFFF >> e;
      if ((a.v & m) == 0)
         return af;
      if (a.v >> 31)
         a.v += m;
      a.v &= ~m;
   } else {
      if (a.v >> 31 == 0)
         a.v = 0;
      else if (a.v << 1)
         a.v = 0xBF800000; // return -1.0f
   }
   return from_softfloat32(a);
}

template<bool ArithNan>
inline float _eosio_f32_trunc( float af ) {
   float32_t a = to_softfloat32(af);
   int e = (int)(a.v >> 23 & 0xff) - 0x7f + 9;
   uint32_t m;
   if (ArithNan && is_nan(a)) {
      return from_softfloat32(propagate_nan<ArithNan>(a));
   }
   if (e >= 23 + 9)
      return af;
   if (e < 9)
      e = 1;
   m = -1U >> e;
   if ((a.v & m) == 0)
      return af;
   a.v &= ~m;
   return from_softfloat32(a);
}

template<bool ArithNan>
inline float _eosio_f32_nearest( float af ) {
   float32_t a = to_softfloat32(af);
   int e = a.v>>23 & 0xff;
   int s = a.v>>31;
   float32_t y;
   if (ArithNan && is_nan(a)) {
      return from_softfloat32(propagate_nan<ArithNan>(a));
   }
   if (e >= 0x7f+23)
      return af;
   if (s)
      y = ::f32_add( ::f32_sub( a, float32_t{inv_float_eps} ), float32_t{inv_float_eps} );
   else
      y = ::f32_sub( ::f32_add( a, float32_t{inv_float_eps} ), float32_t{inv_float_eps} );
   if (::f32_eq( y, {0} ) )
      return s ? -0.0f : 0.0f;
   return from_softfloat32(y);
}

inline bool _eosio_f32_eq( float a, float b ) {  return ::f32_eq( to_softfloat32(a), to_softfloat32(b) ); }
inline bool _eosio_f32_ne( float a, float b ) { return !::f32_eq( to_softfloat32(a), to_softfloat32(b) ); }
inline bool _eosio_f32_lt( float a, float b ) { return ::f32_lt( to_softfloat32(a), to_softfloat32(b) ); }
inline bool _eosio_f32_le( float a, float b ) { return ::f32_le( to_softfloat32(a), to_softfloat32(b) ); }
inline bool _eosio_f32_gt( float af, float bf ) {
   float32_t a = to_softfloat32(af);
   float32_t b = to_softfloat32(bf);
   if (is_nan(a))
      return false;
   if (is_nan(b))
      return false;
   return !::f32_le( a, b );
}

inline bool _eosio_f32_ge( float af, float bf ) {
   float32_t a = to_softfloat32(af);
   float32_t b = to_softfloat32(bf);
   if (is_nan(a))
      return false;
   if (is_nan(b))
      return false;
   return !::f32_lt( a, b );
}

inline double _eosio_f64_add( double a, double b ) {
   float64_t ret = ::f64_add( to_softfloat64(a), to_softfloat64(b) );
   return from_softfloat64(ret);
}

inline double _eosio_f64_sub( double a, double b ) {
   float64_t ret = ::f64_sub( to_softfloat64(a), to_softfloat64(b) );
   return from_softfloat64(ret);
}

inline double _eosio_f64_div( double a, double b ) {
   float64_t ret = ::f64_div( to_softfloat64(a), to_softfloat64(b) );
   return from_softfloat64(ret);
}

inline double _eosio_f64_mul( double a, double b ) {
   float64_t ret = ::f64_mul( to_softfloat64(a), to_softfloat64(b) );
   return from_softfloat64(ret);
}

template<bool ArithNan>
inline double _eosio_f64_min( double af, double bf ) {
   float64_t a = to_softfloat64(af);
   float64_t b = to_softfloat64(bf);
   if (is_nan(a)) {
      return from_softfloat64(propagate_nan<ArithNan>(a));
   }
   if (is_nan(b)) {
      return from_softfloat64(propagate_nan<ArithNan>(b));
   }
   if (f64_sign_bit(a) != f64_sign_bit(b))
      return f64_sign_bit(a) ? af : bf;
   return ::f64_lt( a, b ) ? af : bf;
}

template<bool ArithNan>
inline double _eosio_f64_max( double af, double bf ) {
   float64_t a = to_softfloat64(af);
   float64_t b = to_softfloat64(bf);
   if (is_nan(a)) {
      return from_softfloat64(propagate_nan<ArithNan>(a));
   }
   if (is_nan(b)) {
      return from_softfloat64(propagate_nan<ArithNan>(b));
   }
   if (f64_sign_bit(a) != f64_sign_bit(b))
      return f64_sign_bit(a) ? bf : af;
   return ::f64_lt( a, b ) ? bf : af;
}

inline double _eosio_f64_copysign( double af, double bf ) {
   float64_t a = to_softfloat64(af);
   float64_t b = to_softfloat64(bf);
   a.v &= ~(uint64_t(1) << 63);     // clear the sign bit
   a.v = a.v | ((b.v >> 63) << 63); // add the sign of b
   return from_softfloat64(a);
}

inline double _eosio_f64_abs( double af ) {
   float64_t a = to_softfloat64(af);
   a.v &= ~(uint64_t(1) << 63);
   return from_softfloat64(a);
}

inline double _eosio_f64_neg( double af ) {
   float64_t a = to_softfloat64(af);
   uint64_t sign = a.v >> 63;
   a.v &= ~(uint64_t(1) << 63);
   a.v |= (uint64_t(!sign) << 63);
   return from_softfloat64(a);
}

inline double _eosio_f64_sqrt( double a ) {
   float64_t ret = ::f64_sqrt( to_softfloat64(a) );
   return from_softfloat64(ret);
}

template<bool ArithNan>
inline double _eosio_f64_ceil( double af ) {
   float64_t a = to_softfloat64( af );
   float64_t ret;
   int e = a.v >> 52 & 0x7ff;
   float64_t y;
   if (ArithNan && is_nan(a)) {
      return from_softfloat64(propagate_nan<ArithNan>(a));
   }
   if (e >= 0x3ff+52 || ::f64_eq( a, { 0 } ))
      return af;
   /* y = int(x) - x, where int(x) is an integer neighbor of x */
   if (a.v >> 63)
      y = ::f64_sub( ::f64_add( ::f64_sub( a, float64_t{inv_double_eps} ), float64_t{inv_double_eps} ), a );
   else
      y = ::f64_sub( ::f64_sub( ::f64_add( a, float64_t{inv_double_eps} ), float64_t{inv_double_eps} ), a );
   /* special case because of non-nearest rounding modes */
   if (e <= 0x3ff-1) {
      return a.v >> 63 ? -0.0 : 1.0; //float64_t{0x8000000000000000} : float64_t{0xBE99999A3F800000}; //either -0.0 or 1
   }
   if (::f64_lt( y, to_softfloat64(0) )) {
      ret = ::f64_add( ::f64_add( a, y ), to_softfloat64(1) ); // 0xBE99999A3F800000 } ); // plus 1
      return from_softfloat64(ret);
   }
   ret = ::f64_add( a, y );
   return from_softfloat64(ret);
}

template<bool ArithNan>
inline double _eosio_f64_floor( double af ) {
   float64_t a = to_softfloat64( af );
   float64_t ret;
   int e = a.v >> 52 & 0x7FF;
   float64_t y;
   if (ArithNan && is_nan(a)) {
      return from_softfloat64(propagate_nan<ArithNan>(a));
   }
   if ( a.v == 0x8000000000000000) {
      return af;
   }
   if (e >= 0x3FF+52 || a.v == 0) {
      return af;
   }
   if (a.v >> 63)
      y = ::f64_sub( ::f64_add( ::f64_sub( a, float64_t{inv_double_eps} ), float64_t{inv_double_eps} ), a );
   else
      y = ::f64_sub( ::f64_sub( ::f64_add( a, float64_t{inv_double_eps} ), float64_t{inv_double_eps} ), a );
   if (e <= 0x3FF-1) {
      return a.v>>63 ? -1.0 : 0.0; //float64_t{0xBFF0000000000000} : float64_t{0}; // -1 or 0
   }
   if ( !::f64_le( y, float64_t{0} ) ) {
      ret = ::f64_sub( ::f64_add(a,y), to_softfloat64(1.0));
      return from_softfloat64(ret);
   }
   ret = ::f64_add( a, y );
   return from_softfloat64(ret);
}

template<bool ArithNan>
inline double _eosio_f64_trunc( double af ) {
   float64_t a = to_softfloat64( af );
   int e = (int)(a.v >> 52 & 0x7ff) - 0x3ff + 12;
   uint64_t m;
   if (ArithNan && is_nan(a)) {
      return from_softfloat64(propagate_nan<ArithNan>(a));
   }
   if (e >= 52 + 12)
      return af;
   if (e < 12)
      e = 1;
   m = -1ULL >> e;
   if ((a.v & m) == 0)
      return af;
   a.v &= ~m;
   return from_softfloat64(a);
}

template<bool ArithNan>
inline double _eosio_f64_nearest( double af ) {
   float64_t a = to_softfloat64( af );
   int e = (a.v >> 52 & 0x7FF);
   int s = a.v >> 63;
   float64_t y;
   if (ArithNan && is_nan(a)) {
      return from_softfloat64(propagate_nan<ArithNan>(a));
   }
   if ( e >= 0x3FF+52 )
      return af;
   if ( s )
      y = ::f64_add( ::f64_sub( a, float64_t{inv_double_eps} ), float64_t{inv_double_eps} );
   else
      y = ::f64_sub( ::f64_add( a, float64_t{inv_double_eps} ), float64_t{inv_double_eps} );
   if ( ::f64_eq( y, float64_t{0} ) )
      return s ? -0.0 : 0.0;
   return from_softfloat64(y);
}

inline bool _eosio_f64_eq( double a, double b ) { return ::f64_eq( to_softfloat64(a), to_softfloat64(b) ); }
inline bool _eosio_f64_ne( double a, double b ) { return !::f64_eq( to_softfloat64(a), to_softfloat64(b) ); }
inline bool _eosio_f64_lt( double a, double b ) { return ::f64_lt( to_softfloat64(a), to_softfloat64(b) ); }
inline bool _eosio_f64_le( double a, double b ) { return ::f64_le( to_softfloat64(a), to_softfloat64(b) ); }
inline bool _eosio_f64_gt( double af, double bf ) {
   float64_t a = to_softfloat64(af);
   float64_t b = to_softfloat64(bf);
   if (is_nan(a))
      return false;
   if (is_nan(b))
      return false;
   return !::f64_le( a, b );
}

inline bool _eosio_f64_ge( double af, double bf ) {
   float64_t a = to_softfloat64(af);
   float64_t b = to_softfloat64(bf);
   if (is_nan(a))
      return false;
   if (is_nan(b))
      return false;
   return !::f64_lt( a, b );
}

inline double _eosio_f32_promote( float a ) {
   return from_softfloat64(f32_to_f64( to_softfloat32(a)) );
}

inline float _eosio_f64_demote( double a ) {
   return from_softfloat32(f64_to_f32( to_softfloat64(a)) );
}

inline int32_t _eosio_f32_trunc_i32s( float af ) {
   float32_t a = to_softfloat32(af);
   EOS_VM_ASSERT(!(_eosio_f32_ge(af, 2147483648.0f) || _eosio_f32_lt(af, -2147483648.0f)), wasm_interpreter_exception, "Error, f32.convert_s/i32 overflow" );

   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f32.convert_s/i32 unrepresentable");
   return f32_to_i32( to_softfloat32(_eosio_f32_trunc<false>( af )), 0, false );
}

template<bool FixedBound>
inline int32_t _eosio_f64_trunc_i32s( double af ) {
   float64_t a = to_softfloat64(af);
   if constexpr (FixedBound) {
      EOS_VM_ASSERT(!(_eosio_f64_ge(af, 2147483648.0) || _eosio_f64_le(af, -2147483649.0)), wasm_interpreter_exception, "Error, f64.convert_s/i32 overflow");
   } else {
      EOS_VM_ASSERT(!(_eosio_f64_ge(af, 2147483648.0) || _eosio_f64_lt(af, -2147483648.0)), wasm_interpreter_exception, "Error, f64.convert_s/i32 overflow");
   }
   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f64.convert_s/i32 unrepresentable");
   return f64_to_i32( to_softfloat64(_eosio_f64_trunc<false>( af )), 0, false );
}

inline uint32_t _eosio_f32_trunc_i32u( float af ) {
   float32_t a = to_softfloat32(af);
   EOS_VM_ASSERT(!(_eosio_f32_ge(af, 4294967296.0f) || _eosio_f32_le(af, -1.0f)),wasm_interpreter_exception, "Error, f32.convert_u/i32 overflow");
   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f32.convert_u/i32 unrepresentable");
   return f32_to_ui32( to_softfloat32(_eosio_f32_trunc<false>( af )), 0, false );
}

inline uint32_t _eosio_f64_trunc_i32u( double af ) {
   float64_t a = to_softfloat64(af);
   EOS_VM_ASSERT(!(_eosio_f64_ge(af, 4294967296.0) || _eosio_f64_le(af, -1.0)), wasm_interpreter_exception, "Error, f64.convert_u/i32 overflow");
   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f64.convert_u/i32 unrepresentable");
   return f64_to_ui32( to_softfloat64(_eosio_f64_trunc<false>( af )), 0, false );
}

inline int64_t _eosio_f32_trunc_i64s( float af ) {
   float32_t a = to_softfloat32(af);
      EOS_VM_ASSERT(!(_eosio_f32_ge(af, 9223372036854775808.0f) || _eosio_f32_lt(af, -9223372036854775808.0f)), wasm_interpreter_exception, "Error, f32.convert_s/i64 overflow");
   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f32.convert_s/i64 unrepresentable");
   return f32_to_i64( to_softfloat32(_eosio_f32_trunc<false>( af )), 0, false );
}

inline int64_t _eosio_f64_trunc_i64s( double af ) {
   float64_t a = to_softfloat64(af);
   EOS_VM_ASSERT(!(_eosio_f64_ge(af, 9223372036854775808.0) || _eosio_f64_lt(af, -9223372036854775808.0)), wasm_interpreter_exception, "Error, f64.convert_s/i64 overflow");
   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f64.convert_s/i64 unrepresentable");

   return f64_to_i64( to_softfloat64(_eosio_f64_trunc<false>( af )), 0, false );
}

inline uint64_t _eosio_f32_trunc_i64u( float af ) {
   float32_t a = to_softfloat32(af);
   EOS_VM_ASSERT(!(_eosio_f32_ge(af, 18446744073709551616.0f) || _eosio_f32_le(af, -1.0f)), wasm_interpreter_exception, "Error, f32.convert_u/i64 overflow");
   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f32.convert_u/i64 unrepresentable");
   return f32_to_ui64( to_softfloat32(_eosio_f32_trunc<false>( af )), 0, false );
}

inline uint64_t _eosio_f64_trunc_i64u( double af ) {
   float64_t a = to_softfloat64(af);
   EOS_VM_ASSERT(!(_eosio_f64_ge(af, 18446744073709551616.0) || _eosio_f64_le(af, -1.0)), wasm_interpreter_exception, "Error, f64.convert_u/i64 overflow");
   EOS_VM_ASSERT(!is_nan(a), wasm_interpreter_exception, "Error, f64.convert_u/i64 unrepresentable");
   return f64_to_ui64( to_softfloat64(_eosio_f64_trunc<false>( af )), 0, false );
}

inline float _eosio_i32_to_f32( int32_t a ) {
   return from_softfloat32(i32_to_f32( a ));
}

inline float _eosio_i64_to_f32( int64_t a ) {
   return from_softfloat32(i64_to_f32( a ));
}

inline float _eosio_ui32_to_f32( uint32_t a ) {
   return from_softfloat32(ui32_to_f32( a ));
}

inline float _eosio_ui64_to_f32( uint64_t a ) {
   return from_softfloat32(ui64_to_f32( a ));
}

inline double _eosio_i32_to_f64( int32_t a ) {
   return from_softfloat64(i32_to_f64( a ));
}

inline double _eosio_i64_to_f64( int64_t a ) {
   return from_softfloat64(i64_to_f64( a ));
}

inline double _eosio_ui32_to_f64( uint32_t a ) {
   return from_softfloat64(ui32_to_f64( a ));
}

inline double _eosio_ui64_to_f64( uint64_t a ) {
   return from_softfloat64(ui64_to_f64( a ));
}

inline int32_t _eosio_i32_trunc_sat_f32_s( float af ){
   float32_t a = to_softfloat32(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f32_ge(af, 2147483648.0f)) {
      return INT32_MAX;
   }
   if(_eosio_f32_lt(af, -2147483648.0f)) {
      return INT32_MIN;
   }
   return f32_to_i32_r_minMag(a, false);
}

inline uint32_t _eosio_i32_trunc_sat_f32_u( float af ){
   float32_t a = to_softfloat32(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f32_ge(af, 4294967296.0f)) {
      return UINT32_MAX;
   }
   if(_eosio_f32_le(af, -1.0f)) {
      return 0;
   }
   return f32_to_ui32_r_minMag(a, false);
}

inline int32_t _eosio_i32_trunc_sat_f64_s( double af ){
   float64_t a = to_softfloat64(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f64_ge(af, 2147483648.0f)) {
      return INT32_MAX;
   }
   if(_eosio_f64_lt(af, -2147483648.0f)) {
      return INT32_MIN;
   }
   return f64_to_i32_r_minMag(a, false);
}

inline uint32_t _eosio_i32_trunc_sat_f64_u( double af ){
   float64_t a = to_softfloat64(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f64_ge(af, 4294967296.0f)) {
      return UINT32_MAX;
   }
   if(_eosio_f64_le(af, -1.0f)) {
      return 0;
   }
   return f64_to_ui32_r_minMag(a, false);
}

inline int64_t _eosio_i64_trunc_sat_f32_s( float af ){
   float32_t a = to_softfloat32(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f32_ge(af, 0x1p63f)) {
      return INT64_MAX;
   }
   if(_eosio_f32_lt(af, -0x1p63f)) {
      return INT64_MIN;
   }
   return f32_to_i64_r_minMag(a, false);
}

inline uint64_t _eosio_i64_trunc_sat_f32_u( float af ){
   float32_t a = to_softfloat32(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f32_ge(af, 0x1p64f)) {
      return UINT64_MAX;
   }
   if(_eosio_f32_le(af, -1.0f)) {
      return 0;
   }
   return f32_to_ui64_r_minMag(a, false);
}

inline int64_t _eosio_i64_trunc_sat_f64_s( double af ){
   float64_t a = to_softfloat64(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f64_ge(af, 0x1p63)) {
      return INT64_MAX;
   }
   if(_eosio_f64_lt(af, -0x1p63)) {
      return INT64_MIN;
   }
   return f64_to_i64_r_minMag(a, false);
}

inline uint64_t _eosio_i64_trunc_sat_f64_u( double af ){
   float64_t a = to_softfloat64(af);
   if(is_nan(a)) {
      return 0;
   }
   if(_eosio_f64_ge(af, 0x1p64)) {
      return UINT64_MAX;
   }
   if(_eosio_f64_le(af, -1.0)) {
      return 0;
   }
   return f64_to_ui64_r_minMag(a, false);
}

template<typename F, typename R, typename... T>
auto get_funptr(F f, R (F::*)(T...) const) {
   return static_cast<R(*)(T...)>(f);
}
template<typename F>
auto get_funptr(F f) { return get_funptr(f, &F::operator()); }

template<typename R, typename T>
inline v128_t apply_f32x4(v128_t a, R(*f)(T)) {
   T a_[4];
   R result_[4];
   static_assert(sizeof(a_) == sizeof(a));
   std::memcpy(&a_, &a, sizeof(a));
   for(int i = 0; i < 4; ++i) {
      result_[i] = f(a_[i]);
   }
   v128_t result;
   static_assert(sizeof(result_) == sizeof(result));
   memcpy(&result, &result_, sizeof(result));
   return result;
}

template<typename R, typename T>
inline v128_t apply_f32x4(v128_t a, v128_t b, R (*f)(T,T)) {
   T a_[4];
   T b_[4];
   R result_[4];
   static_assert(sizeof(a_) == sizeof(a));
   std::memcpy(&a_, &a, sizeof(a));
   std::memcpy(&b_, &b, sizeof(b));
   for(int i = 0; i < 4; ++i) {
      result_[i] = f(a_[i], b_[i]);
   }
   v128_t result;
   static_assert(sizeof(result_) == sizeof(result));
   memcpy(&result, &result_, sizeof(result));
   return result;
}

template<typename F>
inline v128_t apply_f32x4(v128_t a, F f) {
   return apply_f32x4(a, get_funptr(f));
}

template<typename F>
inline v128_t apply_f32x4(v128_t a, v128_t b, F f) {
   return apply_f32x4(a, b, get_funptr(f));
}

inline v128_t _eosio_f32x4_eq(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) -> std::int32_t {
      return ::f32_eq(a, b)?-1:0;
   });
}

inline v128_t _eosio_f32x4_ne(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) -> std::int32_t {
      return !::f32_eq(a,b)?-1:0;
   });
}

inline v128_t _eosio_f32x4_lt(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) -> std::int32_t {
      return ::f32_lt(a,b)?-1:0;
   });
}

inline v128_t _eosio_f32x4_gt(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) -> std::int32_t {
      return ::f32_lt(b,a)?-1:0;
   });
}

inline v128_t _eosio_f32x4_le(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) -> std::int32_t {
      return ::f32_le(a,b)?-1:0;
   });
}

inline v128_t _eosio_f32x4_ge(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) -> std::int32_t {
      return ::f32_le(b,a)?-1:0;
   });
}

inline v128_t _eosio_f32x4_ceil(v128_t a) {
   return apply_f32x4(a, _eosio_f32_ceil<true>);
}

inline v128_t _eosio_f32x4_floor(v128_t a) {
   return apply_f32x4(a, _eosio_f32_floor<true>);
}

inline v128_t _eosio_f32x4_trunc(v128_t a) {
   return apply_f32x4(a, _eosio_f32_trunc<true>);
}

inline v128_t _eosio_f32x4_nearest(v128_t a) {
   return apply_f32x4(a, _eosio_f32_nearest<true>);
}

inline v128_t _eosio_f32x4_abs(v128_t a) {
   return apply_f32x4(a, _eosio_f32_abs);
}

inline v128_t _eosio_f32x4_neg(v128_t a) {
   return apply_f32x4(a, _eosio_f32_neg);
}

inline v128_t _eosio_f32x4_sqrt(v128_t a) {
   return apply_f32x4(a, _eosio_f32_sqrt);
}

inline v128_t _eosio_f32x4_add(v128_t a, v128_t b) {
   return apply_f32x4(a, b, ::f32_add);
}

inline v128_t _eosio_f32x4_sub(v128_t a, v128_t b) {
   return apply_f32x4(a, b, ::f32_sub);
}

inline v128_t _eosio_f32x4_mul(v128_t a, v128_t b) {
   return apply_f32x4(a, b, ::f32_mul);
}

inline v128_t _eosio_f32x4_div(v128_t a, v128_t b) {
   return apply_f32x4(a, b, ::f32_div);
}

inline v128_t _eosio_f32x4_min(v128_t a, v128_t b) {
   return apply_f32x4(a, b, _eosio_f32_min<true>);
}

inline v128_t _eosio_f32x4_max(v128_t a, v128_t b) {
   return apply_f32x4(a, b, _eosio_f32_max<true>);
}

inline v128_t _eosio_f32x4_pmin(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) { return ::f32_lt(b, a)?b:a; });
}

inline v128_t _eosio_f32x4_pmax(v128_t a, v128_t b) {
   return apply_f32x4(a, b, [](float32_t a, float32_t b) { return ::f32_lt(a, b)?b:a; });
}

template<typename R, typename T>
inline v128_t apply_f64x2(v128_t a, R(*f)(T)) {
   T a_[2];
   R result_[2];
   static_assert(sizeof(a_) == sizeof(a));
   std::memcpy(&a_, &a, sizeof(a));
   for(int i = 0; i < 2; ++i) {
      result_[i] = f(a_[i]);
   }
   v128_t result;
   static_assert(sizeof(result_) == sizeof(result));
   memcpy(&result, &result_, sizeof(result));
   return result;
}

template<typename R, typename T>
inline v128_t apply_f64x2(v128_t a, v128_t b, R (*f)(T,T)) {
   T a_[2];
   T b_[2];
   R result_[2];
   static_assert(sizeof(a_) == sizeof(a));
   std::memcpy(&a_, &a, sizeof(a));
   std::memcpy(&b_, &b, sizeof(b));
   for(int i = 0; i < 2; ++i) {
      result_[i] = f(a_[i], b_[i]);
   }
   v128_t result;
   static_assert(sizeof(result_) == sizeof(result));
   memcpy(&result, &result_, sizeof(result));
   return result;
}

template<typename F>
inline v128_t apply_f64x2(v128_t a, F f) {
   return apply_f64x2(a, get_funptr(f));
}

template<typename F>
inline v128_t apply_f64x2(v128_t a, v128_t b, F f) {
   return apply_f64x2(a, b, get_funptr(f));
}

inline v128_t _eosio_f64x2_eq(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) -> std::int64_t {
      return ::f64_eq(a, b)?-1:0;
   });
}

inline v128_t _eosio_f64x2_ne(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) -> std::int64_t {
      return !::f64_eq(a,b)?-1:0;
   });
}

inline v128_t _eosio_f64x2_lt(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) -> std::int64_t {
      return ::f64_lt(a,b)?-1:0;
   });
}

inline v128_t _eosio_f64x2_gt(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) -> std::int64_t {
      return ::f64_lt(b,a)?-1:0;
   });
}

inline v128_t _eosio_f64x2_le(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) -> std::int64_t {
      return ::f64_le(a,b)?-1:0;
   });
}

inline v128_t _eosio_f64x2_ge(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) -> std::int64_t {
      return ::f64_le(b,a)?-1:0;
   });
}

inline v128_t _eosio_f64x2_ceil(v128_t a) {
   return apply_f64x2(a, _eosio_f64_ceil<true>);
}

inline v128_t _eosio_f64x2_floor(v128_t a) {
   return apply_f64x2(a, _eosio_f64_floor<true>);
}

inline v128_t _eosio_f64x2_trunc(v128_t a) {
   return apply_f64x2(a, _eosio_f64_trunc<true>);
}

inline v128_t _eosio_f64x2_nearest(v128_t a) {
   return apply_f64x2(a, _eosio_f64_nearest<true>);
}

inline v128_t _eosio_f64x2_abs(v128_t a) {
   return apply_f64x2(a, _eosio_f64_abs);
}

inline v128_t _eosio_f64x2_neg(v128_t a) {
   return apply_f64x2(a, _eosio_f64_neg);
}

inline v128_t _eosio_f64x2_sqrt(v128_t a) {
   return apply_f64x2(a, _eosio_f64_sqrt);
}

inline v128_t _eosio_f64x2_add(v128_t a, v128_t b) {
   return apply_f64x2(a, b, ::f64_add);
}

inline v128_t _eosio_f64x2_sub(v128_t a, v128_t b) {
   return apply_f64x2(a, b, ::f64_sub);
}

inline v128_t _eosio_f64x2_mul(v128_t a, v128_t b) {
   return apply_f64x2(a, b, ::f64_mul);
}

inline v128_t _eosio_f64x2_div(v128_t a, v128_t b) {
   return apply_f64x2(a, b, ::f64_div);
}

inline v128_t _eosio_f64x2_min(v128_t a, v128_t b) {
   return apply_f64x2(a, b, _eosio_f64_min<true>);
}

inline v128_t _eosio_f64x2_max(v128_t a, v128_t b) {
   return apply_f64x2(a, b, _eosio_f64_max<true>);
}

inline v128_t _eosio_f64x2_pmin(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) { return ::f64_lt(b, a)?b:a; });
}

inline v128_t _eosio_f64x2_pmax(v128_t a, v128_t b) {
   return apply_f64x2(a, b, [](float64_t a, float64_t b) { return ::f64_lt(a, b)?b:a; });
}

inline v128_t _eosio_i32x4_trunc_sat_f32x4_s(v128_t a) {
   return apply_f32x4(a, _eosio_i32_trunc_sat_f32_s);
}

inline v128_t _eosio_i32x4_trunc_sat_f32x4_u(v128_t a) {
   return apply_f32x4(a, _eosio_i32_trunc_sat_f32_u);
}

inline v128_t _eosio_f32x4_convert_i32x4_s(v128_t a) {
   return apply_f32x4(a, _eosio_i32_to_f32);
}

inline v128_t _eosio_f32x4_convert_i32x4_u(v128_t a) {
   return apply_f32x4(a, _eosio_ui32_to_f32);
}

template<typename R, typename T>
inline v128_t apply_f64x2_zero(v128_t a, R(*f)(T)) {
   T a_[2];
   R result_[4];
   static_assert(sizeof(a_) == sizeof(a));
   std::memcpy(&a_, &a, sizeof(a));
   for(int i = 0; i < 2; ++i) {
      result_[i] = f(a_[i]);
   }
   result_[2] = result_[3] = 0;
   v128_t result;
   static_assert(sizeof(result_) == sizeof(result));
   memcpy(&result, &result_, sizeof(result));
   return result;
}

template<typename R, typename T>
inline v128_t apply_f32x4_low(v128_t a, R(*f)(T)) {
   T a_[4];
   R result_[2];
   static_assert(sizeof(a_) == sizeof(a));
   std::memcpy(&a_, &a, sizeof(a));
   for(int i = 0; i < 2; ++i) {
      result_[i] = f(a_[i]);
   }
   v128_t result;
   static_assert(sizeof(result_) == sizeof(result));
   memcpy(&result, &result_, sizeof(result));
   return result;
}

inline v128_t _eosio_i32x4_trunc_sat_f64x2_s_zero(v128_t a) {
   return apply_f64x2_zero(a, _eosio_i32_trunc_sat_f64_s);
}

inline v128_t _eosio_i32x4_trunc_sat_f64x2_u_zero(v128_t a) {
   return apply_f64x2_zero(a, _eosio_i32_trunc_sat_f64_u);
}

inline v128_t _eosio_f64x2_convert_low_i32x4_s(v128_t a) {
   return apply_f32x4_low(a, _eosio_i32_to_f64);
}

inline v128_t _eosio_f64x2_convert_low_i32x4_u(v128_t a) {
   return apply_f32x4_low(a, _eosio_ui32_to_f64);
}

inline v128_t _eosio_f32x4_demote_f64x2_zero(v128_t a) {
   return apply_f64x2_zero(a, _eosio_f64_demote);
}

inline v128_t _eosio_f64x2_promote_low_f32x4(v128_t a) {
   return apply_f32x4_low(a, _eosio_f32_promote);
}

}} //ns eosio::vm
#endif
