using System;
using System.Numerics;

using Lingvo.PosTagger.Tensors.Cpu;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tensors
{
    public static class TensorApplyCPU
    {
        #region [.Tensor iteration methods.]
        unsafe public delegate void Apply1KernelFunction( float* x );
        unsafe public delegate void Apply2KernelFunction( float* x, float* y );
        unsafe public delegate void Apply3KernelFunction( float* x, float* y, float* z );
        unsafe public delegate void Apply4KernelFunction( float* x, float* y, float* z, float* k );
        unsafe public delegate void Apply5KernelFunction( float* x, float* y, float* z, float* k, float* l );
        unsafe public delegate void ApplyDim2KernelFuncton( float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY );
        unsafe public delegate void ApplyDim3KernelFuncton( float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY, float* z, long sizeZ, long stridesZ );

        unsafe static void Apply1( Tensor tensor1, Apply1KernelFunction func )
        {
            float* buffer1 = (float*) CpuNativeHelpers.GetBufferStart( tensor1 );

            TensorIterState tensor1Iter = new TensorIterState( buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides );

            do
            {
                for ( ; !tensor1Iter.ReachedBlockEnd(); tensor1Iter.BlockStep() )
                {
                    func( tensor1Iter.Data );
                }
            } 
            while ( tensor1Iter.NextBlock() );
        }

        unsafe static void Apply2( Tensor tensor1, Tensor tensor2, Apply2KernelFunction func, int step = 1 )
        {
            float* buffer1 = (float*) CpuNativeHelpers.GetBufferStart( tensor1 );
            float* buffer2 = (float*) CpuNativeHelpers.GetBufferStart( tensor2 );

            TensorIterState tensor1Iter = new TensorIterState( buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides, step );
            TensorIterState tensor2Iter = new TensorIterState( buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides, step );

            do
            {
                for ( ; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd(); tensor1Iter.BlockStep(), tensor2Iter.BlockStep() )
                {
                    func( tensor1Iter.Data, tensor2Iter.Data );
                }
            } 
            while ( tensor1Iter.NextBlock() && tensor2Iter.NextBlock() );
        }

        unsafe static void Apply3( Tensor tensor1, Tensor tensor2, Tensor tensor3, Apply3KernelFunction func, int step = 1 )
        {
            var buffer1 = (float*) CpuNativeHelpers.GetBufferStart( tensor1 );
            var buffer2 = (float*) CpuNativeHelpers.GetBufferStart( tensor2 );
            var buffer3 = (float*) CpuNativeHelpers.GetBufferStart( tensor3 );

            TensorIterState tensor1Iter = new TensorIterState( buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides, step );
            TensorIterState tensor2Iter = new TensorIterState( buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides, step );
            TensorIterState tensor3Iter = new TensorIterState( buffer3, tensor3.DimensionCount, tensor3.Sizes, tensor3.Strides, step );

            do
            {
                for ( ; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd();
                        tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep() )
                {
                    func( tensor1Iter.Data, tensor2Iter.Data, tensor3Iter.Data );
                }
            }
            while ( tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() );
        }

        unsafe static void Apply4( Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, Apply4KernelFunction func )
        {
            var buffer1 = (float*) CpuNativeHelpers.GetBufferStart( tensor1 );
            var buffer2 = (float*) CpuNativeHelpers.GetBufferStart( tensor2 );
            var buffer3 = (float*) CpuNativeHelpers.GetBufferStart( tensor3 );
            var buffer4 = (float*) CpuNativeHelpers.GetBufferStart( tensor4 );

            TensorIterState tensor1Iter = new TensorIterState( buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides );
            TensorIterState tensor2Iter = new TensorIterState( buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides );
            TensorIterState tensor3Iter = new TensorIterState( buffer3, tensor3.DimensionCount, tensor3.Sizes, tensor3.Strides );
            TensorIterState tensor4Iter = new TensorIterState( buffer4, tensor4.DimensionCount, tensor4.Sizes, tensor4.Strides );

            do
            {
                for ( ; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd() && !tensor4Iter.ReachedBlockEnd();
                    tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep(), tensor4Iter.BlockStep() )
                {
                    func( tensor1Iter.Data, tensor2Iter.Data, tensor3Iter.Data, tensor4Iter.Data );
                }
            } 
            while ( tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() && tensor4Iter.NextBlock() );
        }

        unsafe static void Apply5( Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, Tensor tensor5, Apply5KernelFunction func, int step = 1 )
        {
            var buffer1 = (float*) CpuNativeHelpers.GetBufferStart( tensor1 );
            var buffer2 = (float*) CpuNativeHelpers.GetBufferStart( tensor2 );
            var buffer3 = (float*) CpuNativeHelpers.GetBufferStart( tensor3 );
            var buffer4 = (float*) CpuNativeHelpers.GetBufferStart( tensor4 );
            var buffer5 = (float*) CpuNativeHelpers.GetBufferStart( tensor5 );

            TensorIterState tensor1Iter = new TensorIterState( buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides, step );
            TensorIterState tensor2Iter = new TensorIterState( buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides, step );
            TensorIterState tensor3Iter = new TensorIterState( buffer3, tensor3.DimensionCount, tensor3.Sizes, tensor3.Strides, step );
            TensorIterState tensor4Iter = new TensorIterState( buffer4, tensor4.DimensionCount, tensor4.Sizes, tensor4.Strides, step );
            TensorIterState tensor5Iter = new TensorIterState( buffer5, tensor5.DimensionCount, tensor5.Sizes, tensor5.Strides, step );

            do
            {
                for ( ; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd() && !tensor4Iter.ReachedBlockEnd() && !tensor5Iter.ReachedBlockEnd();
                    tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep(), tensor4Iter.BlockStep(), tensor5Iter.BlockStep() )
                {
                    func( tensor1Iter.Data, tensor2Iter.Data, tensor3Iter.Data, tensor4Iter.Data, tensor5Iter.Data );
                }
            } 
            while ( tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() && tensor4Iter.NextBlock() && tensor5Iter.NextBlock() );
        }

        unsafe static void ApplyDim2( Tensor tensor1, Tensor tensor2, int iterationDim, ApplyDim2KernelFuncton func )
        {
            var buffer1 = (float*) CpuNativeHelpers.GetBufferStart( tensor1 );
            var buffer2 = (float*) CpuNativeHelpers.GetBufferStart( tensor2 );

            TensorDimIterState tensor1Iter = new TensorDimIterState( buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides, iterationDim );
            TensorDimIterState tensor2Iter = new TensorDimIterState( buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides, iterationDim );

            do
            {
                func( tensor1Iter.Data, tensor1Iter.Size, tensor1Iter.Stride, tensor2Iter.Data, tensor2Iter.Size, tensor2Iter.Stride );
            } 
            while ( tensor1Iter.NextBlock() && tensor2Iter.NextBlock() );
        }

        unsafe static void ApplyDim3( Tensor tensor1, Tensor tensor2, Tensor tensor3, int iterationDim, ApplyDim3KernelFuncton func )
        {
            var buffer1 = (float*) CpuNativeHelpers.GetBufferStart( tensor1 );
            var buffer2 = (float*) CpuNativeHelpers.GetBufferStart( tensor2 );
            var buffer3 = (float*) CpuNativeHelpers.GetBufferStart( tensor3 );

            TensorDimIterState tensor1Iter = new TensorDimIterState( buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides, iterationDim );
            TensorDimIterState tensor2Iter = new TensorDimIterState( buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides, iterationDim );
            TensorDimIterState tensor3Iter = new TensorDimIterState( buffer3, tensor3.DimensionCount, tensor3.Sizes, tensor3.Strides, iterationDim );

            do
            {
                func( tensor1Iter.Data, tensor1Iter.Size, tensor1Iter.Stride,
                      tensor2Iter.Data, tensor2Iter.Size, tensor2Iter.Stride,
                      tensor3Iter.Data, tensor3Iter.Size, tensor3Iter.Stride );
            } 
            while ( tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() );
        }
        #endregion

        unsafe public static void Gather( Tensor result, Tensor src, int dim, Tensor indices )
        {
            unsafe void func( float* rData, long rSize, long rStride, float* sData, long sSize, long sStride, float* iData, long iSize, long iStride )
            {
                for ( int i = 0; i < iSize; ++i )
                {
                    long idx = (long) *(iData + i * iStride);
                    if ( idx < 0 || idx >= sSize ) { throw new IndexOutOfRangeException( $"Invalid index in gather. Idx = '{idx}', sSize = '{sSize}'." ); }

                    *(rData + i * rStride) = sData[ idx * sStride ];
                }
            }

            ApplyDim3( result, src, indices, dim, func );
        }

        unsafe public static void Scatter( Tensor result, Tensor src, int dim, Tensor indices )
        {
            unsafe void func( float* rData, long rSize, long rStride, float* sData, long sSize, long sStride, float* iData, long iSize, long iStride )
            {
                for ( int i = 0; i < iSize; ++i )
                {
                    long idx = (long) *(iData + i * iStride);
                    if ( idx < 0 || idx >= rSize ) { throw new IndexOutOfRangeException( $"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize}'." ); }

                    rData[ idx * rStride ] = *(sData + i * sStride);
                }
            }

            ApplyDim3( result, src, indices, dim, func );
        }

        unsafe public static void ScatterAdd( Tensor result, Tensor src, int dim, Tensor indices )
        {
            unsafe void func( float* rData, long rSize, long rStride, float* sData, long sSize, long sStride, float* iData, long iSize, long iStride )
            {
                for ( int i = 0; i < iSize; ++i )
                {
                    long idx = (long) *(iData + i * iStride);
                    if ( idx < 0 || idx >= rSize ) { throw new IndexOutOfRangeException( $"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize}'." ); }

                    rData[ idx * rStride ] += *(sData + i * sStride);
                }
            }

            ApplyDim3( result, src, indices, dim, func );
        }

        unsafe public static void ScatterFill( Tensor result, float value, int dim, Tensor indices )
        {
            unsafe void func( float* rData, long rSize, long rStride, float* iData, long iSize, long iStride )
            {
                for ( int i = 0; i < iSize; ++i )
                {
                    long idx = (long) *(iData + i * iStride);
                    if ( idx < 0 || idx >= rSize ) { throw new IndexOutOfRangeException( $"Invalid index in ScatterFill. Idx = '{idx}', rSize = '{rSize}'" ); }

                    rData[ idx * rStride ] = value;
                }
            }

            ApplyDim2( result, indices, dim, func );
        }

        unsafe public static void Fill( Tensor result, float value )
        {
            unsafe void func( float* r ) => *r = value;
            Apply1( result, func );
        }

        unsafe public static void Clamp( Tensor result, Tensor src, float min, float max )
        {
            unsafe void func( float* r, float* s ) => *r = clamp( *s, min, max );
            Apply2( result, src, func );
        }

        unsafe public static void Copy( Tensor result, Tensor src )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && src.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* s )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanS = new Span<float>( s, vectorSize );
                    var vecS = new Vector<float>( spanS );
                    vecS.CopyTo( spanR );
                }

                Apply2( result, src, funcVec, vectorSize );
            }
            else
            {
                unsafe static void func( float* r, float* s ) => *r = *s;
                Apply2( result, src, func );
            }
        }

        unsafe public static void Sum( Tensor result, Tensor src, int dimension )
        {
            unsafe void func( float* r, long rSize, long rStride, float* s, long sSize, long sStride )
            {
                float sum = 0.0f;
                for ( long i = 0; i < sSize; ++i )
                {
                    sum += s[ i * sStride ];
                }
                *r = sum;
            }
            ApplyDim2( result, src, dimension, func );
        }

        unsafe public static void Argmax( Tensor resultIndices, Tensor src, int dimension )
        {

            unsafe void func( float* rIndVal, long rIndSize, long rIndStride, float* s, long sSize, long sStride )
            {
                float value = s[ 0 ];
                float index = 0;
                for ( long i = 1; i < sSize; ++i )
                {
                    float currentVal = s[ i * sStride ];
                    if ( currentVal > value )
                    {
                        value = currentVal;
                        index = (float) i;
                    }
                }
                *rIndVal = index;
            }

            ApplyDim2( resultIndices, src, dimension, func );
        }

        unsafe public static void Max( Tensor result, Tensor src, int dimension )
        {
            unsafe void func( float* r, long rSize, long rStride, float* s, long sSize, long sStride )
            {
                float value = s[ 0 ];
                for ( long i = 1; i < sSize; ++i )
                {
                    value = Math.Max( value, s[ i * sStride ] );
                }
                *r = value;
            }

            ApplyDim2( result, src, dimension, func );
        }

        unsafe public static void Add( Tensor result, Tensor lhs, Tensor rhs )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && lhs.Strides[ ^1 ] == 1 && rhs.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* left, float* right )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanLeft = new Span<float>( left, vectorSize );
                    var spanRight = new Span<float>( right, vectorSize );

                    var vecLeft = new Vector<float>( spanLeft );
                    var vecRight = new Vector<float>( spanRight );

                    var vecR = vecLeft + vecRight;
                    vecR.CopyTo( spanR );
                }

                Apply3( result, lhs, rhs, funcVec, vectorSize );
            }
            else
            {
                unsafe void func( float* r, float* left, float* right ) => *r = add( *left, *right );
                Apply3( result, lhs, rhs, func );
            }
        }

        unsafe public static void Sub( Tensor result, Tensor lhs, Tensor rhs )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && lhs.Strides[ ^1 ] == 1 && rhs.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* left, float* right )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanLeft = new Span<float>( left, vectorSize );
                    var spanRight = new Span<float>( right, vectorSize );

                    var vecLeft = new Vector<float>( spanLeft );
                    var vecRight = new Vector<float>( spanRight );

                    var vecR = vecLeft - vecRight;
                    vecR.CopyTo( spanR );
                }

                Apply3( result, lhs, rhs, funcVec, vectorSize );
            }
            else
            {
                unsafe void func( float* r, float* left, float* right ) => *r = *left - *right;
                Apply3( result, lhs, rhs, func );
            }
        }

        unsafe public static void Add( Tensor result, Tensor src, float value )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && src.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                var vecV = new Vector<float>( value );
                unsafe void funcVec( float* r, float* s )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanS = new Span<float>( s, vectorSize );

                    var vecS = new Vector<float>( spanS );

                    var vecR = vecS + vecV;
                    vecR.CopyTo( spanR );
                }

                Apply2( result, src, funcVec, vectorSize );
            }
            else
            {
                unsafe void func( float* r, float* s ) => *r = add( *s, value );
                Apply2( result, src, func );
            }
        }

        unsafe public static void Pow( Tensor result, Tensor src, float value )
        {
            unsafe void func( float* r, float* s ) => *r = (float) Math.Pow( *s, value );
            Apply2( result, src, func );
        }

        unsafe public static void RSub( Tensor result, float value, Tensor src )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && src.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                var vecV = new Vector<float>( value );
                unsafe void funcVec( float* r, float* s )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanS = new Span<float>( s, vectorSize );

                    var vecS = new Vector<float>( spanS );

                    var vecR = vecV - vecS;
                    vecR.CopyTo( spanR );
                }

                Apply2( result, src, funcVec, vectorSize );
            }
            else
            {
                unsafe void func( float* r, float* s ) => *r = value - *s;
                Apply2( result, src, func );
            }
        }

        unsafe public static void Mul( Tensor result, Tensor src, float value )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && src.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                var vecV = new Vector<float>( value );
                unsafe void funcVec( float* r, float* s )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanS = new Span<float>( s, vectorSize );

                    var vecS = new Vector<float>( spanS );

                    var vecR = vecS * vecV;
                    vecR.CopyTo( spanR );
                }

                Apply2( result, src, funcVec, vectorSize );
            }
            else
            {
                unsafe void func( float* r, float* s ) => *r = mul( *s, value );
                Apply2( result, src, func );
            }
        }

        unsafe public static void Div( Tensor result, Tensor lhs, float rhs )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && lhs.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                var vecV = new Vector<float>( rhs );
                unsafe void funcVec( float* r, float* s )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanS = new Span<float>( s, vectorSize );

                    var vecS = new Vector<float>( spanS );

                    var vecR = vecS / vecV;
                    vecR.CopyTo( spanR );
                }

                Apply2( result, lhs, funcVec, vectorSize );
            }
            else
            {
                unsafe void func( float* r, float* s ) => *r = div( *s, rhs );
                Apply2( result, lhs, func );
            }
        }

        unsafe public static void Mul( Tensor result, Tensor lhs, Tensor rhs )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && lhs.Strides[ ^1 ] == 1 && rhs.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* left, float* right )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanLeft = new Span<float>( left, vectorSize );
                    var spanRight = new Span<float>( right, vectorSize );

                    var vecLeft = new Vector<float>( spanLeft );
                    var vecRight = new Vector<float>( spanRight );

                    var vecR = vecLeft * vecRight;
                    vecR.CopyTo( spanR );
                }

                Apply3( result, lhs, rhs, funcVec, vectorSize );
            }
            else
            {
                unsafe static void func( float* r, float* left, float* right ) => *r = mul( *left, *right );
                Apply3( result, lhs, rhs, func );
            }
        }

        unsafe public static void Div( Tensor result, Tensor lhs, Tensor rhs )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && lhs.Strides[ ^1 ] == 1 && rhs.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* left, float* right )
                {
                    var spanR     = new Span<float>( r, vectorSize );
                    var spanLeft  = new Span<float>( left, vectorSize );
                    var spanRight = new Span<float>( right, vectorSize );

                    var vecLeft  = new Vector<float>( spanLeft );
                    var vecRight = new Vector<float>( spanRight );

                    var vecR = vecLeft / vecRight;
                    vecR.CopyTo( spanR );
                }

                Apply3( result, lhs, rhs, funcVec, vectorSize );
            }
            else
            {
                unsafe static void func( float* r, float* left, float* right ) => *r = div( *left, *right );
                Apply3( result, lhs, rhs, func );
            }
        }

        unsafe static public void Relu( Tensor result, Tensor src )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && src.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* s )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanS = new Span<float>( s, vectorSize );

                    var vecS = new Vector<float>( spanS );

                    var vecR = Vector.Max( vecS, Vector<float>.Zero );
                    vecR.CopyTo( spanR );
                }

                Apply2( result, src, funcVec, vectorSize );
            }
            else
            {
                unsafe static void func( float* r, float* s ) => *r = relu( *s );
                Apply2( result, src, func );
            }
        }

        unsafe static public void Rsqrt( Tensor result, Tensor src )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && src.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* s )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanS = new Span<float>( s, vectorSize );

                    var vecS = new Vector<float>( spanS );

                    var vecR = Vector<float>.One / Vector.SquareRoot( vecS );
                    vecR.CopyTo( spanR );
                }

                Apply2( result, src, funcVec, vectorSize );
            }
            else
            {
                unsafe static void func( float* r, float* s ) => *r = (float) (1.0 / Math.Sqrt( *s ));
                Apply2( result, src, func );
            }
        }

        unsafe static public void Sigmoid( Tensor result, Tensor src )
        {
            unsafe static void func( float* r, float* s ) => *r = sigmoid( *s );
            Apply2( result, src, func );
        }

        unsafe static public void SigmoidD( Tensor result, Tensor resW, Tensor resG )
        {
            unsafe static void func( float* r, float* x, float* y ) => *r = sigmoidD( *x, *y );
            Apply3( result, resW, resG, func );
        }

        unsafe static public void Tanh( Tensor result, Tensor src )
        {
            unsafe static void func( float* r, float* s ) => *r = (float) Math.Tanh( *s );
            Apply2( result, src, func );
        }

        unsafe static public void Log( Tensor result, Tensor src )
        {
            unsafe static void func( float* r, float* s ) => *r = (float) Math.Log( *s );
            Apply2( result, src, func );
        }

        unsafe static public void TanhD( Tensor result, Tensor resW, Tensor resG )
        {
            unsafe static void func( float* r, float* x, float* y ) => *r = tanhD( *x, *y );
            Apply3( result, resW, resG, func );
        }

        unsafe static public void AddTanh( Tensor result, Tensor srcX, Tensor srcY )
        {
            unsafe static void func( float* r, float* x, float* y ) => *r = addtanh( *x, *y );
            Apply3( result, srcX, srcY, func );
        }

        unsafe static public void AddTanhD( Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ )
        {
            unsafe static void func( float* r, float* x, float* y, float* z ) => *r = addtanhD( *x, *y, *z );
            Apply4( result, srcX, srcY, srcZ, func );
        }

        unsafe static public void ReluD( Tensor result, Tensor srcW, Tensor srcG )
        {
            unsafe static void func( float* r, float* y, float* x ) => *r = relud( *y, *x );
            Apply3( result, srcW, srcG, func );
        }

        unsafe static public void AddMulV( Tensor result, Tensor srcX, Tensor srcY, float val )
        {
            unsafe void func( float* r, float* x, float* y ) => *r = *x + (*y * val);
            Apply3( result, srcX, srcY, func );
        }

        unsafe static public void MulMulAdd( Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ, Tensor srcW )
        {
            int vectorSize = Vector<float>.Count;
            if ( result.Strides[ ^1 ] == 1 && srcX.Strides[ ^1 ] == 1 && srcY.Strides[ ^1 ] == 1 && srcZ.Strides[ ^1 ] == 1 && srcW.Strides[ ^1 ] == 1 && result.Sizes[ ^1 ] % vectorSize == 0 )
            {
                unsafe void funcVec( float* r, float* x, float* y, float* z, float* w )
                {
                    var spanR = new Span<float>( r, vectorSize );
                    var spanX = new Span<float>( x, vectorSize );
                    var spanY = new Span<float>( y, vectorSize );
                    var spanZ = new Span<float>( z, vectorSize );
                    var spanW = new Span<float>( w, vectorSize );

                    var vecX = new Vector<float>( spanX );
                    var vecY = new Vector<float>( spanY );
                    var vecZ = new Vector<float>( spanZ );
                    var vecW = new Vector<float>( spanW );

                    var vecR = vecX * vecY + vecZ * vecW;
                    vecR.CopyTo( spanR );
                }

                Apply5( result, srcX, srcY, srcZ, srcW, funcVec, vectorSize );
            }
            else
            {
                unsafe static void func( float* r, float* x, float* y, float* z, float* w ) => *r = mulmuladd( *x, *y, *z, *w );
                Apply5( result, srcX, srcY, srcZ, srcW, func );
            }
        }

        unsafe static public void AddMul( Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ )
        {
            unsafe static void func( float* r, float* x, float* y, float* z ) => *r = addmul( *x, *y, *z );
            Apply4( result, srcX, srcY, srcZ, func );
        }

        unsafe static public void AddDiv( Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ )
        {
            unsafe static void func( float* r, float* x, float* y, float* z ) => *r = adddiv( *x, *y, *z );
            Apply4( result, srcX, srcY, srcZ, func );
        }

        unsafe static public void BuildSelfMask( Tensor result, Tensor originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue )
        {
            var ptResult          = (float*) CpuNativeHelpers.GetBufferStart( result );
            var ptOriginalLengths = (float*) CpuNativeHelpers.GetBufferStart( originalLengths );

            for ( int j = 0; j < rows; j++ )
            {
                float* resultRow = ptResult + j * cols;
                int batchIdx = j / paddedSeqLen;
                int seqIdxInBatch = j % paddedSeqLen;

                for ( int id = 0; id < cols; id++ )
                {
                    int originalLength = (int) ptOriginalLengths[ batchIdx ];
                    if ( id < originalLength && seqIdxInBatch < originalLength )
                    {
                        resultRow[ id ] = value;
                    }
                    else
                    {
                        resultRow[ id ] = maskedValue;
                    }
                }
            }
        }

        unsafe static public void BuildTriMask( Tensor result, int rows, int cols, float value, float maskedValue )
        {
            var ptResult = (float*) CpuNativeHelpers.GetBufferStart( result );

            for ( int j = 0; j < rows; j++ )
            {
                float* resultRow = ptResult + j * cols;
                for ( int id = 0; id < cols; id++ )
                {

                    if ( id <= j )
                    {
                        resultRow[ id ] = value;
                    }
                    else
                    {
                        resultRow[ id ] = maskedValue;
                    }
                }
            }
        }

        unsafe static public void BuildSelfTriMask( Tensor result, Tensor originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue )
        {
            var ptResult          = (float*) CpuNativeHelpers.GetBufferStart( result );
            var ptOriginalLengths = (float*) CpuNativeHelpers.GetBufferStart( originalLengths );

            for ( int j = 0; j < rows; j++ )
            {
                float* resultRow = ptResult + j * cols;
                int batchIdx = j / paddedSeqLen;
                int seqIdxInBatch = j % paddedSeqLen;

                for ( int id = 0; id < cols; id++ )
                {
                    int originalLength = (int) ptOriginalLengths[ batchIdx ];
                    if ( id < originalLength && seqIdxInBatch < originalLength && id <= seqIdxInBatch )
                    {
                        resultRow[ id ] = value;
                    }
                    else
                    {
                        resultRow[ id ] = maskedValue;
                    }

                }
            }
        }

        unsafe static public void BuildSrcTgtMask( Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int rows, int cols, int tgtPaddedSeqLen, float value, float maskedValue )
        {
            var ptResult             = (float*) CpuNativeHelpers.GetBufferStart( result );
            var ptSrcOriginalLengths = (float*) CpuNativeHelpers.GetBufferStart( srcOriginalLengths );
            var ptTgtOriginalLengths = (float*) CpuNativeHelpers.GetBufferStart( tgtOriginalLengths );

            for ( int j = 0; j < rows; j++ )
            {
                float* resultRow = ptResult + j * cols;
                int batchIdx = j / tgtPaddedSeqLen;
                int seqIdxInBatch = j % tgtPaddedSeqLen;

                for ( int id = 0; id < cols; id++ )
                {
                    int srcOriginalLength = (int) ptSrcOriginalLengths[ batchIdx ];
                    int tgtOriginalLength = (int) ptTgtOriginalLengths[ batchIdx ];

                    if ( id < srcOriginalLength && seqIdxInBatch < tgtOriginalLength )
                    {
                        resultRow[ id ] = value;
                    }
                    else
                    {
                        resultRow[ id ] = maskedValue;
                    }
                }
            }
        }
        unsafe static public void Softmax( Tensor tOut, Tensor tIn, int rows, int cols )
        {
            var pOut = (float*) CpuNativeHelpers.GetBufferStart( tOut );
            var pIn  = (float*) CpuNativeHelpers.GetBufferStart( tIn );

            for ( int j = 0; j < rows; ++j )
            {
                float* so = pOut + j * cols;
                float* sp = pIn + j * cols;

                float max = sp[ 0 ];
                for ( int i = 1; i < cols; ++i )
                    max = Math.Max( max, sp[ i ] );

                float sum = 0.0f;
                for ( int i = 0; i < cols; ++i )
                {
                    float ex = (float) Math.Exp( sp[ i ] - max );
                    so[ i ] = ex;
                    sum += ex;
                }

                Span<float> spanSO = new Span<float>( so, cols );
                int vectorSize = Vector<float>.Count;
                int k = 0;
                Vector<float> vecSum = new Vector<float>( sum );
                for ( k = 0; k < cols - vectorSize; k += vectorSize )
                {
                    Vector<float> vecSO = new Vector<float>( spanSO.Slice( k ) );
                    vecSO /= vecSum;

                    vecSO.CopyTo( spanSO.Slice( k ) );
                }
                for ( ; k < cols; k++ )
                {
                    so[ k ] /= sum;
                }
            }
        }

        unsafe static public void SoftmaxGrad( Tensor grad_, Tensor adj_, Tensor val_, int rows, int cols, bool addGrad )
        {

            var grad = (float*) CpuNativeHelpers.GetBufferStart( grad_ );
            var adj = (float*) CpuNativeHelpers.GetBufferStart( adj_ );
            var val = (float*) CpuNativeHelpers.GetBufferStart( val_ );

            for ( int j = 0; j < rows; ++j )
            {
                float* gradRow = grad + j * cols;
                float* adjRow  = adj + j * cols;
                float* valRow  = val + j * cols;

                float sum = 0.0f;
                for ( int i = 0; i < cols; ++i )
                {
                    sum += valRow[ i ] * adjRow[ i ];
                }

                for ( int i = 0; i < cols; ++i )
                {
                    if ( addGrad )
                    {
                        gradRow[ i ] += valRow[ i ] * (adjRow[ i ] - sum);
                    }
                    else
                    {
                        gradRow[ i ] = valRow[ i ] * (adjRow[ i ] - sum);
                    }
                }
            }
        }

        unsafe static public void IndexSelect( Tensor result_, Tensor src_, Tensor indice_, int rows, int cols )
        {
            var result = (float*) CpuNativeHelpers.GetBufferStart( result_ );
            var src    = (float*) CpuNativeHelpers.GetBufferStart( src_ );
            var indice = (float*) CpuNativeHelpers.GetBufferStart( indice_ );

            for ( int j = 0; j < rows; j++ )
            {

                int srcIdx = (int) indice[ j ];
                float* resultRow = result + j * cols;
                float* srcRow = src + srcIdx * cols;

                for ( int i = 0; i < cols; ++i )
                {
                    resultRow[ i ] = srcRow[ i ];
                }
            }
        }

        unsafe static public void IndexSelectGrad( Tensor grad_, Tensor adj_, Tensor indice_, int rows, int cols )
        {
            var grad   = (float*) CpuNativeHelpers.GetBufferStart( grad_ );
            var adj    = (float*) CpuNativeHelpers.GetBufferStart( adj_ );
            var indice = (float*) CpuNativeHelpers.GetBufferStart( indice_ );

            for ( int j = 0; j < rows; j++ )
            {

                int gradIdx = (int) indice[ j ];
                float* adjRow = adj + j * cols;
                float* gradRow = grad + gradIdx * cols;

                for ( int i = 0; i < cols; ++i )
                {
                    gradRow[ i ] += adjRow[ i ];
                }
            }
        }

        unsafe static public void LayerNorm( Tensor out_, Tensor in_, Tensor gamma_, Tensor beta_, float eps, int rows, int cols )
        {
            var outPtr = (float*) CpuNativeHelpers.GetBufferStart( out_ );
            var inPtr  = (float*) CpuNativeHelpers.GetBufferStart( in_ );
            var alpha  = (float*) CpuNativeHelpers.GetBufferStart( gamma_ );
            var beta   = (beta_ != null) ? (float*) CpuNativeHelpers.GetBufferStart( beta_ ) : null;

            for ( int j = 0; j < rows; ++j )
            {
                float* so = outPtr + j * cols;
                float* sp = inPtr + j * cols;

                Span<float> spanSP = new Span<float>( sp, cols );

                float sum = 0.0f;
                int vectorSize = Vector<float>.Count;
                Vector<float> vecAdded = Vector<float>.Zero;
                int i = 0;

                for ( i = 0; i < cols - vectorSize; i += vectorSize )
                {
                    Vector<float> vecSp = new Vector<float>( spanSP.Slice( i ) );
                    vecAdded += vecSp;
                }
                sum = Vector.Dot( vecAdded, Vector<float>.One );
                for ( ; i < cols; i++ )
                {
                    sum += sp[ i ];
                }

                float mean = sum / cols;
                float sqSum = 0.0f;

                Vector<float> vecMean = new Vector<float>( mean );
                for ( i = 0; i < cols - vectorSize; i += vectorSize )
                {
                    Vector<float> vecSp = new Vector<float>( spanSP.Slice( i ) );
                    Vector<float> vecEx = vecSp - vecMean;
                    sqSum += Vector.Dot( vecEx, vecEx );
                }
                for ( ; i < cols; ++i )
                {
                    float ex = sp[ i ] - mean;
                    sqSum += ex * ex;
                }

                var sigma = (float) Math.Sqrt( eps + sqSum / cols );

                var spanSO    = new Span<float>( so, cols );
                var spanAlpha = new Span<float>( alpha, cols );
                var spanBeta  = (beta != null) ? new Span<float>( beta, cols ) : null;
                var vecSigma  = new Vector<float>( sigma );

                for ( i = 0; i < cols - vectorSize; i += vectorSize )
                {
                    var vecSp    = new Vector<float>( spanSP.Slice( i ) );
                    var vecAlpha = new Vector<float>( spanAlpha.Slice( i ) );

                    var vecT = vecAlpha * ((vecSp - vecMean) / vecSigma);

                    if ( spanBeta != null )
                    {
                        var vecBeta = new Vector<float>( spanBeta.Slice( i ) );
                        vecT += vecBeta;
                    }

                    vecT.CopyTo( spanSO.Slice( i ) );
                }
                for ( ; i < cols; ++i )
                {
                    float t = alpha[ i ] * ((sp[ i ] - mean) / sigma);
                    if ( beta != null )
                    {
                        t += beta[ i ];
                    }

                    so[ i ] = t;
                }
            }
        }

        unsafe static public void LayerNormGrad( Tensor gradX_, Tensor gradGamma_, Tensor gradBeta_, Tensor adj_, Tensor y_, Tensor x_, Tensor gamma_, Tensor beta_, int rows, int cols, float eps )
        {
            var gradX     = (float*) CpuNativeHelpers.GetBufferStart( gradX_ );
            var gradGamma = (float*) CpuNativeHelpers.GetBufferStart( gradGamma_ );
            var gradBeta  = gradBeta_ != null ? (float*) CpuNativeHelpers.GetBufferStart( gradBeta_ ) : null;
            var adj       = (float*) CpuNativeHelpers.GetBufferStart( adj_ );
            var y         = (float*) CpuNativeHelpers.GetBufferStart( y_ );
            var x         = (float*) CpuNativeHelpers.GetBufferStart( x_ );
            var gamma     = (float*) CpuNativeHelpers.GetBufferStart( gamma_ );
            var beta      = (beta_ != null) ? (float*) CpuNativeHelpers.GetBufferStart( beta_ ) : null;

            if ( beta != null )
            {
                for ( int j = 0; j < rows; ++j )
                {
                    float* xRow = x + j * cols;
                    float* yRow = y + j * cols;
                    float* adjRow = adj + j * cols;
                    float* gradXRow = gradX + j * cols;

                    float sum_x = 0.0f;
                    float sum_adj = 0.0f;
                    float sum_adj_x = 0.0f;
                    float sum_sqr = 0.0f;

                    for ( int i = 0; i < cols; ++i )
                    {
                        sum_x += xRow[ i ];
                        sum_adj_x += adjRow[ i ] * (yRow[ i ] - (beta != null ? beta[ i ] : 0.0f)) / gamma[ i ];
                        sum_adj += adjRow[ i ];
                    }

                    float mean = sum_x / cols;
                    for ( int i = 0; i < cols; ++i )
                    {
                        float ex = xRow[ i ] - mean;
                        sum_sqr += ex * ex;
                    }

                    float sigma = (float) Math.Sqrt( eps + sum_sqr / cols );
                    for ( int i = 0; i < cols; ++i )
                    {
                        float grad_x = 0.0f;
                        float x_hat = (yRow[ i ] - beta[ i ]) / gamma[ i ];
                        grad_x += cols * adjRow[ i ];
                        grad_x -= sum_adj;
                        grad_x -= sum_adj_x * x_hat;
                        grad_x /= cols * sigma;

                        gradXRow[ i ] += gamma[ i ] * grad_x;
                        gradGamma[ i ] += adjRow[ i ] * x_hat;
                        gradBeta[ i ] += adjRow[ i ];
                    }
                }
            }
            else
            {
                for ( int j = 0; j < rows; ++j )
                {
                    float* xRow = x + j * cols;
                    float* yRow = y + j * cols;
                    float* adjRow = adj + j * cols;
                    float* gradXRow = gradX + j * cols;

                    float sum_x = 0.0f;
                    float sum_adj = 0.0f;
                    float sum_adj_x = 0.0f;
                    float sum_sqr = 0.0f;

                    for ( int i = 0; i < cols; ++i )
                    {
                        sum_x += xRow[ i ];
                        sum_adj_x += adjRow[ i ] * (yRow[ i ] - (beta != null ? beta[ i ] : 0.0f)) / gamma[ i ];
                        sum_adj += adjRow[ i ];
                    }

                    float mean = sum_x / cols;

                    for ( int i = 0; i < cols; ++i )
                    {
                        float ex = xRow[ i ] - mean;
                        sum_sqr += ex * ex;
                    }

                    float sigma = (float) Math.Sqrt( eps + sum_sqr / cols );

                    for ( int i = 0; i < cols; ++i )
                    {
                        float grad_x = 0.0f;
                        float x_hat = yRow[ i ] / gamma[ i ];
                        grad_x += cols * adjRow[ i ];
                        grad_x -= sum_adj;
                        grad_x -= sum_adj_x * x_hat;
                        grad_x /= cols * sigma;

                        gradXRow[ i ] += gamma[ i ] * grad_x;
                        gradGamma[ i ] += adjRow[ i ] * x_hat;
                    }
                }
            }
        }

        unsafe static public void Adam( Tensor tw, Tensor tg, Tensor tv, Tensor tm, int rows, int cols, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps )
        {
            var w = (float*) CpuNativeHelpers.GetBufferStart( tw );
            var g = (float*) CpuNativeHelpers.GetBufferStart( tg );
            var v = (float*) CpuNativeHelpers.GetBufferStart( tv );
            var m = (float*) CpuNativeHelpers.GetBufferStart( tm );

            for ( int j = 0; j < rows; j++ )
            {
                float* sw = w + j * cols;
                float* sg = g + j * cols;
                float* sv = v + j * cols;
                float* sm = m + j * cols;

                for ( int i = 0; i < cols; i++ )
                {
                    if ( sg[ i ] != 0.0 )
                    {
                        float g2 = sg[ i ] / batchSize;

                        if ( g2 > clipval )
                        {
                            g2 = clipval;
                        }
                        if ( g2 < -clipval )
                        {
                            g2 = -clipval;
                        }

                        sm[ i ] = sm[ i ] * decay_rate_m + (1.0f - decay_rate_m) * g2;
                        sv[ i ] = sv[ i ] * decay_rate_v + (1.0f - decay_rate_v) * g2 * g2;

                        double m_cap = sm[ i ] / (1.0 - Math.Pow( decay_rate_m, iter ));
                        double v_cap = sv[ i ] / (1.0 - Math.Pow( decay_rate_v, iter ));

                        sw[ i ] -= (float) (step_size * m_cap / (Math.Sqrt( v_cap ) + eps));

                        sg[ i ] = 0;
                    }
                }
            }
        }

        #region [.Internal operations.]
        [M(O.AggressiveInlining)] private static float relu( float w )
        {
            if ( w < 0.0f )
                return 0.0f;
            return w;

        }
        [M(O.AggressiveInlining)] private static float relud( float w, float g )
        {
            if ( w > 0.0f )
                return g;
            return 0.0f;
        }
        [M(O.AggressiveInlining)] private static float add( float x, float y ) => x + y;
        [M(O.AggressiveInlining)] private static float mul( float x, float y ) => x * y;        
        [M(O.AggressiveInlining)] private static float div( float x, float y ) => x / y;        
        [M(O.AggressiveInlining)] private static float sigmoid( float x ) => (float) (1.0 / (1.0 + Math.Exp( -x )));
        [M(O.AggressiveInlining)] private static float sigmoidD( float resW, float resG ) => resW * (1.0f - resW) * resG;
        [M(O.AggressiveInlining)] private static float mulmuladd( float x, float y, float z, float w ) => x * y + z * w;        
        [M(O.AggressiveInlining)] private static float clamp( float val, float min, float max )
        {
            if ( val < min )
                return min;
            if ( val > max )
                return max;
            return val;
        }
        [M(O.AggressiveInlining)] private static float addmul( float x, float y, float z ) => x + y * z;
        [M(O.AggressiveInlining)] private static float adddiv( float x, float y, float z ) => x + y / z;        
        [M(O.AggressiveInlining)] private static float addtanh( float x, float y ) => (float) Math.Tanh( x + y );
        [M(O.AggressiveInlining)] private static float addtanhD( float t, float resW, float resG ) => t + (1.0f - resW * resW) * resG;
        [M(O.AggressiveInlining)] static float tanhD( float resW, float resG ) => (1.0f - resW * resW) * resG;
        #endregion
    }
}
