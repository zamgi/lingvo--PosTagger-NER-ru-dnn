namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    public class CudaAllocator : IAllocator
    {
        private readonly TSCudaContext _Context;
        private readonly int _DeviceId;

        public CudaAllocator( TSCudaContext context, int deviceId )
        {
            _Context  = context;
            _DeviceId = deviceId;
        }

        public TSCudaContext Context => _Context;
        public int DeviceId => _DeviceId;

        public Storage Allocate( DType elementType, long elementCount ) => new CudaStorage( this, _Context, _Context.CudaContextForDevice( _DeviceId ), elementType, elementCount );
        public float GetAllocatedMemoryRatio() => Context.AllocatorForDevice( DeviceId ).GetAllocatedMemoryRatio();
    }
}
