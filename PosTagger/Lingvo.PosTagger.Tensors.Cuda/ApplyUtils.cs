using System;

using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    public static class ApplyUtils
    {
        private const int APPLY_THREADS_PER_BLOCK = 32 * 16;

        public static dim3 GetApplyBlock() => new dim3( APPLY_THREADS_PER_BLOCK );

        // returns Ceil(x / y)
        public static long CeilDiv( long x, long y ) => (x + y - 1) / y;

        public static dim3 GetApplyGrid( CudaDeviceProperties deviceInfo, long totalElements )
        {
            int smCount = deviceInfo.MultiProcessorCount;

            // Rationale for grid size - from cuTorch source code:
            // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
            // which seems to be a good sweetspot for latency hiding
            int maxSize = 4 * smCount;
            long targetSize = CeilDiv( totalElements, APPLY_THREADS_PER_BLOCK );
            return new dim3( (uint) Math.Min( targetSize, maxSize ) );
        }

        public static bool CanUse32BitIndexMath( Tensor tensor )
        {
            long elements = tensor.ElementCount();
            if ( elements >= uint.MaxValue )
            {
                return (false);
            }

            long offset = 0;
            long linearId = elements - 1;

            for ( int i = tensor.DimensionCount - 1; i >= 0; --i )
            {
                long curDimIndex = linearId % tensor.Sizes[ i ];
                long curDimOffset = curDimIndex * tensor.Strides[ i ];
                offset += curDimOffset;
                linearId /= tensor.Sizes[ i ];
            }

            if ( offset >= uint.MaxValue )
            {
                return (false);
            }
            return (true);
        }
    }
}
