namespace Lingvo.PosTagger.Tensors.Cpu
{
    /// <summary>
    /// 
    /// </summary>
    public class CpuAllocator : IAllocator
    {
        public CpuAllocator() { }
        public Storage Allocate( DType elementType, long elementCount ) => new CpuStorage( this, elementType, elementCount );
        public float GetAllocatedMemoryRatio() => 0.0f;
    }
}
