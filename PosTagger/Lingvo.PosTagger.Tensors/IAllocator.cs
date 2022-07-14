namespace Lingvo.PosTagger.Tensors
{
    public interface IAllocator
    {
        Storage Allocate( DType elementType, long elementCount );
        float GetAllocatedMemoryRatio();
    }
}
