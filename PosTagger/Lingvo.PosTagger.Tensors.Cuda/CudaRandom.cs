using Lingvo.PosTagger.Tensors.Cpu;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// Basic implementation of random ops for CUDA. All we do here is generate the tensors on the
    /// CPU then copy to the CUDA buffer. This is definitely not an optimal implementation.
    /// </summary>
    [OpsClass]
    public class CudaRandom
    {
        private readonly CpuAllocator _CpuAllocator;
        private readonly CpuRandom    _CpuRandom;

        public CudaRandom()
        {
            _CpuAllocator = new CpuAllocator();
            _CpuRandom    = new CpuRandom();
        }


        [RegisterOpStorageType("random_uniform", typeof(CudaStorage))]
        public void Uniform( Tensor result, int? seed, float min, float max )
        {
            using var cpuCopy = new Tensor( _CpuAllocator, result.ElementType, result.Sizes );
            _CpuRandom.Uniform( cpuCopy, seed, min, max );
            Ops.Copy( result, cpuCopy );
        }

        [RegisterOpStorageType("random_normal", typeof(CudaStorage))]
        public void Normal( Tensor result, int? seed, float mean, float stdv )
        {
            using var cpuCopy = new Tensor( _CpuAllocator, result.ElementType, result.Sizes );
            _CpuRandom.Normal( cpuCopy, seed, mean, stdv );
            Ops.Copy( result, cpuCopy );
        }

        [RegisterOpStorageType("random_exponential", typeof(CudaStorage))]
        public void Exponential( Tensor result, int? seed, float lambda )
        {
            using var cpuCopy = new Tensor( _CpuAllocator, result.ElementType, result.Sizes );
            _CpuRandom.Exponential( cpuCopy, seed, lambda );
            Ops.Copy( result, cpuCopy );
        }

        [RegisterOpStorageType("random_cauchy", typeof(CudaStorage))]
        public void Cauchy( Tensor result, int? seed, float median, float sigma )
        {
            using var cpuCopy = new Tensor( _CpuAllocator, result.ElementType, result.Sizes );
            _CpuRandom.Cauchy( cpuCopy, seed, median, sigma );
            Ops.Copy( result, cpuCopy );
        }

        [RegisterOpStorageType("random_lognormal", typeof(CudaStorage))]
        public void LogNormal( Tensor result, int? seed, float mean, float stdv )
        {
            using var cpuCopy = new Tensor( _CpuAllocator, result.ElementType, result.Sizes );
            _CpuRandom.LogNormal( cpuCopy, seed, mean, stdv );
            Ops.Copy( result, cpuCopy );
        }

        [RegisterOpStorageType("random_geometric", typeof(CudaStorage))]
        public void Geometric( Tensor result, int? seed, float p )
        {
            using var cpuCopy = new Tensor( _CpuAllocator, result.ElementType, result.Sizes );
            _CpuRandom.Geometric( cpuCopy, seed, p );
            Ops.Copy( result, cpuCopy );
        }

        [RegisterOpStorageType("random_bernoulli", typeof(CudaStorage))]
        public void Bernoulli( Tensor result, int? seed, float p )
        {
            using var cpuCopy = new Tensor( _CpuAllocator, result.ElementType, result.Sizes );
            _CpuRandom.Bernoulli( cpuCopy, seed, p );
            Ops.Copy( result, cpuCopy );
        }
    }
}
