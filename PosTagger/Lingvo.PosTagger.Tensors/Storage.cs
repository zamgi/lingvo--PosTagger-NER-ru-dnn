using System;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public abstract class Storage : RefCounted
    {
        public Storage( IAllocator allocator, DType elementType, long elementCount )
        {
            Allocator    = allocator;
            ElementType  = elementType;
            ElementCount = elementCount;
        }

        /// <summary>
        /// Gets a reference to the allocator that constructed this Storage object.
        /// </summary>
        public IAllocator Allocator { get; }
        public DType ElementType { get; }
        public long ElementCount { get; }

        public long ByteLength => ElementCount * ElementType.Size();

        public bool IsOwnerExclusive() => (GetCurrentRefCount() == 1);

        public abstract string LocationDescription();
        public abstract int[] GetElementsAsInt( long index, int length );
        public abstract void SetElementsAsInt( long index, int[] value );
        public abstract float GetElementAsFloat( long index );
        public abstract float[] GetElementsAsFloat( long index, int length );
        public abstract void SetElementAsFloat( long index, float value );
        public abstract void SetElementsAsFloat( long index, float[] value );

        public abstract void CopyToStorage( long storageIndex, IntPtr src, long byteCount );
        public abstract void CopyFromStorage( IntPtr dst, long storageIndex, long byteCount );
    }
}
