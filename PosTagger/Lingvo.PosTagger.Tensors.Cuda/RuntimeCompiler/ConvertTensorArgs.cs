using ManagedCuda;

namespace Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler
{
    /// <summary>
    /// 
    /// </summary>
    public static class ConvertTensorArgs
    {
        /// <summary>
        /// 
        /// </summary>
        private unsafe struct TensorInfoIndex64
        {
            public ulong data;
            public fixed ulong sizes  [ TSCudaContext.MAX_DIMS ];
            public fixed ulong strides[ TSCudaContext.MAX_DIMS ];
            public int dims;
        }
        /// <summary>
        /// 
        /// </summary>
        private unsafe struct TensorInfoIndex32
        {
            public ulong data;
            public fixed uint sizes  [ TSCudaContext.MAX_DIMS ];
            public fixed uint strides[ TSCudaContext.MAX_DIMS ];
            public int dims;
        }

        public static void Convert( CudaContext context, bool index32, object[] args )
        {
            for ( int i = 0; i < args.Length; ++i )
            {
                if ( args[ i ] is Tensor tensor )
                {
                    args[ i ] = MakeTensorInfo( context, tensor, index32 );
                }
            }
        }

        public static unsafe object MakeTensorInfo( CudaContext context, Tensor tensor, bool index32, int flattenDim = -1 )
        {
            if ( index32 )
            {
                var ti = new TensorInfoIndex32()
                {
                    data = CudaHelpers.GetBufferStart( tensor ),
                    dims = tensor.DimensionCount
                };
                for ( int i = 0; i < tensor.DimensionCount; ++i )
                {
                    ti.sizes  [ i ] = (uint) tensor.Sizes  [ i ];
                    ti.strides[ i ] = (uint) tensor.Strides[ i ];
                }

                if ( flattenDim != -1 )
                {
                    ti.sizes[ flattenDim ] = 1;
                }

                return (ti);
            }
            else
            {
                var ti = new TensorInfoIndex64()
                {
                    data = CudaHelpers.GetBufferStart( tensor ),
                    dims = tensor.DimensionCount
                };
                for ( int i = 0; i < tensor.DimensionCount; ++i )
                {
                    ti.sizes  [ i ] = (ulong) tensor.Sizes  [ i ];
                    ti.strides[ i ] = (ulong) tensor.Strides[ i ];
                }

                if ( flattenDim != -1 )
                {
                    ti.sizes[ flattenDim ] = 1;
                }

                return (ti);
            }
        }
    }
}
