using System;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tensors
{
    unsafe public class TensorIterState
    {
        private long[] _Sizes;
        private long[] _Strides;
        private long   _Stride;
        private long   _Size;
        private int    _Dim;
        private long[] _Counter;
        private int    _Step;
        private long   _Index;
        public float*  _Data;
 
        public TensorIterState( float* buffer, int dimCount, long[] sizes, long[] strides, int step = 1 )
        {
            _Sizes = sizes;
            _Strides = strides;
            _Step = step;

            _Index = 0;
            _Data = buffer;

            for ( _Dim = dimCount - 1; _Dim >= 0; _Dim-- )
            {
                if ( sizes[ _Dim ] != 1 )
                    break;
            }

            // Get stride for dimension
            _Stride = (_Dim == -1 ? 0 : strides[ _Dim ]);

            // Find largest contiguous section
            // Note: this updates dim and size
            _Size = 1;
            for ( _Dim = dimCount - 1; _Dim >= 0; _Dim-- )
            {
                if ( strides[ _Dim ] == _Size )
                {
                    _Size *= sizes[ _Dim ];
                }
                else
                {
                    break;
                }
            }

            if ( _Size % step != 0 ) throw (new ArgumentException( $"Size '{_Size}' mod step '{step}' must be zero." ));

            // Counter keeps track of how many iterations have been performed on each dimension
            // that is *not* part of the above contiguous block
            // Iterations are performed from highest dimension index to lowest.
            // When a complete iteration of dimension i is finished, the counter for dim i-1 gets incremented by 1
            _Counter = new long[ _Dim + 1 ];
            for ( int i = 0; i < _Dim + 1; ++i )
            {
                _Counter[ i ] = 0;
            }
        }

        public float* Data { [M(O.AggressiveInlining)] get => _Data; }

        [M(O.AggressiveInlining)] public bool ReachedBlockEnd() => !(_Index < _Size);
        [M(O.AggressiveInlining)] public void BlockStep()
        {
            _Index += _Step;
            _Data += (_Stride * _Step);
        }

        // Returns true if there is another block to iterate over, returns false if we are at end of iteration
        [M(O.AggressiveInlining)] public bool NextBlock()
        {
            // If not at end of current block yet, do nothing
            if ( _Index == _Size )
            {
                // If contiguous block encompassed all dimensions, we are done
                if ( _Dim == -1 )
                    return (false);

                // Reset data offset
                _Data -= _Size * _Stride;

                // Update counter and data for next contiguous block
                for ( long j = _Dim; j >= 0; --j )
                {
                    _Counter[ j ]++;
                    _Data += _Strides[ j ];

                    if ( _Counter[ j ] == _Sizes[ j ] )
                    {
                        if ( j == 0 )
                        {
                            return (false);
                        }
                        else
                        {
                            _Data -= _Counter[ j ] * _Strides[ j ];
                            _Counter[ j ] = 0;
                        }
                    }
                    else
                    {
                        break;
                    }
                }

                _Index = 0;
            }

            return (true);
        }
    }
}
