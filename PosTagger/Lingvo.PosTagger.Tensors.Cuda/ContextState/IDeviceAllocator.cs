using System;

using ManagedCuda.BasicTypes;

namespace Lingvo.PosTagger.Tensors.Cuda.ContextState
{
    /// <summary>
    /// 
    /// </summary>
    public interface IDeviceMemory
    {
        CUdeviceptr Pointer { get; }
        void Free();
    }
    /// <summary>
    /// 
    /// </summary>
    public interface IDeviceAllocator : IDisposable
    {
        IDeviceMemory Allocate( long byteCount );
        float GetAllocatedMemoryRatio();
    }
}
