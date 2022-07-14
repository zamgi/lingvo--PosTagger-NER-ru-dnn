using Lingvo.PosTagger.Tensors.Cuda.DeviceCode;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    [OpsClass]
    public class CudaIndexingOps
    {
        private readonly GatherScatterKernels _Gather = new GatherScatterKernels();
        public CudaIndexingOps() { }

        [RegisterOpStorageType("gather", typeof(CudaStorage))]
        public Tensor Gather( Tensor result, Tensor src, int dimension, Tensor indices ) => _Gather.Gather( result, src, dimension, indices );

        [RegisterOpStorageType("scatter", typeof(CudaStorage))]
        public Tensor Scatter( Tensor result, Tensor src, int dimension, Tensor indices ) => _Gather.Scatter( result, src, dimension, indices );

        [RegisterOpStorageType("scatter_add", typeof(CudaStorage))]
        public Tensor ScatterAdd( Tensor result, Tensor src, int dimension, Tensor indices ) => _Gather.ScatterAdd( result, src, dimension, indices );

        [RegisterOpStorageType("scatter_fill", typeof(CudaStorage))]
        public Tensor ScatterFill( Tensor result, float value, int dimension, Tensor indices ) => _Gather.ScatterFill( result, value, dimension, indices );
    }
}
