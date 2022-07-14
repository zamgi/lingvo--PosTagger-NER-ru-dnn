using Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler;

namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode.Headers
{
    /// <summary>
    /// 
    /// </summary>
    [CudaInclude("Code", "General")]
    public static class KernelGeneral
    {
        public static readonly string Code = @"

#define __int64 long long
#define __int32 int

#define MAX_CUTORCH_DIMS " + TSCudaContext.MAX_DIMS + "\n" + @"

template <typename IndexType>
struct TensorInfo {
  float* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};

";

    }
}
