using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tensors
{
    unsafe public class TensorDimIterState
    {
        private long[] _Sizes;
        private long[] _Strides;
        private int    _DimensionCount;
        private int    _IterationDim;
        private long[] _Counter;
        private long   _Stride;
        private long   _Size;
        private float* _Data;

        unsafe public TensorDimIterState( float* buffer, int dimCount, long[] sizes, long[] strides, int iterationDim )
        {
            _Sizes = sizes;
            _Strides = strides;
            _IterationDim = iterationDim;
            _DimensionCount = dimCount;

            _Data = buffer;

            _Size   = sizes[ iterationDim ];
            _Stride = strides[ iterationDim ];
            
            _Counter = new long[ dimCount ];
            for ( int i = 0; i < dimCount; ++i )
            {
                _Counter[ i ] = 0;
            }
        }

        public long   Stride { [M(O.AggressiveInlining)] get => _Stride; }
        public long   Size   { [M(O.AggressiveInlining)] get => _Size; }
        public float* Data   { [M(O.AggressiveInlining)] get => _Data; }

        // Returns true if there is another block to iterate over,
        // returns false if we are at end of iteration
        public bool NextBlock()
        {
            if ( _DimensionCount == 1 )
            {
                return (false);
            }

            for ( int i = 0; i < _DimensionCount; ++i )
            {
                if ( i == _IterationDim )
                {
                    if ( i == _DimensionCount - 1 )
                    {
                        return (false);
                    }
                    continue;
                }

                _Counter[ i ]++;
                _Data += _Strides[ i ];

                if ( _Counter[ i ] == _Sizes[ i ] )
                {
                    if ( i == _DimensionCount - 1 )
                    {
                        return (false);
                    }
                    else
                    {
                        _Data -= _Counter[ i ] * _Strides[ i ];
                        _Counter[ i ] = 0;
                    }
                }
                else
                {
                    break;
                }
            }

            return (true);
        }
    }
}
