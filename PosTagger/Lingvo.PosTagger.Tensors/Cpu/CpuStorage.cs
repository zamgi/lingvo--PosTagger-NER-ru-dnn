using System;
using System.Runtime.InteropServices;

namespace Lingvo.PosTagger.Tensors.Cpu
{
    /// <summary>
    /// 
    /// </summary>
    public class CpuStorage : Storage
    {
        public IntPtr _Buffer;
        public CpuStorage( IAllocator allocator, DType ElementType, long elementCount ) : base( allocator, ElementType, elementCount ) => _Buffer = Marshal.AllocHGlobal( new IntPtr( ByteLength ) );
        protected override void Destroy()
        {
            Marshal.FreeHGlobal( _Buffer );
            _Buffer = IntPtr.Zero;
        }
        public override string LocationDescription() => "CPU";
        public IntPtr PtrAtElement( long index ) => new IntPtr( _Buffer.ToInt64() + (index * ElementType.Size()) );
        public override float GetElementAsFloat( long index )
        {
            unsafe
            {
                if ( ElementType == DType.Float32 )
                {
                    return ((float*) _Buffer.ToPointer())[ index ];
                }
                else if ( ElementType == DType.Float64 )
                {
                    return (float) ((double*) _Buffer.ToPointer())[ index ];
                }
                else if ( ElementType == DType.Int32 )
                {
                    return ((int*) _Buffer.ToPointer())[ index ];
                }
                else if ( ElementType == DType.UInt8 )
                {
                    return ((byte*) _Buffer.ToPointer())[ index ];
                }
                else
                {
                    throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
                }
            }
        }
        public override float[] GetElementsAsFloat( long index, int length )
        {
            unsafe
            {
                if ( ElementType == DType.Float32 )
                {
                    float* p = ((float*) _Buffer.ToPointer());
                    var array = new float[ length ];
                    for ( int i = 0; i < length; i++ )
                    {
                        array[ i ] = *(p + i);
                    }
                    return (array);
                }
                else
                {
                    throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
                }
            }
        }
        public override int[] GetElementsAsInt( long index, int length )
        {
            unsafe
            {
                if ( ElementType == DType.Int32 )
                {
                    int* p = ((int*) _Buffer.ToPointer());
                    var array = new int[ length ];
                    for ( int i = 0; i < length; i++ )
                    {
                        array[ i ] = *(p + i);
                    }
                    return (array);
                }
                else
                {
                    throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
                }
            }
        }

        public override void SetElementAsFloat( long index, float value )
        {
            unsafe
            {
                if ( ElementType == DType.Float32 )
                {
                    ((float*) _Buffer.ToPointer())[ index ] = value;
                }
                else if ( ElementType == DType.Float64 )
                {
                    ((double*) _Buffer.ToPointer())[ index ] = value;
                }
                else if ( ElementType == DType.Int32 )
                {
                    ((int*) _Buffer.ToPointer())[ index ] = (int) value;
                }
                else if ( ElementType == DType.UInt8 )
                {
                    ((byte*) _Buffer.ToPointer())[ index ] = (byte) value;
                }
                else
                {
                    throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
                }
            }
        }
        public override void SetElementsAsFloat( long index, float[] value )
        {
            unsafe
            {
                if ( ElementType == DType.Float32 )
                {
                    for ( int i = 0; i < value.Length; i++ )
                    {
                        ((float*) _Buffer.ToPointer())[ index + i ] = value[ i ];
                    }
                }
                else
                {
                    throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
                }
            }
        }
        public override void SetElementsAsInt( long index, int[] value )
        {
            unsafe
            {
                if ( ElementType == DType.Int32 )
                {
                    for ( int i = 0; i < value.Length; i++ )
                    {
                        ((int*) _Buffer.ToPointer())[ index + i ] = value[ i ];
                    }
                }
                else
                {
                    throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
                }
            }
        }

        public override void CopyToStorage( long storageIndex, IntPtr src, long byteCount )
        {
            IntPtr dstPtr = PtrAtElement( storageIndex );
            unsafe
            {
                Buffer.MemoryCopy( src.ToPointer(), dstPtr.ToPointer(), byteCount, byteCount );
            }
        }
        public override void CopyFromStorage( IntPtr dst, long storageIndex, long byteCount )
        {
            IntPtr srcPtr = PtrAtElement( storageIndex );
            unsafe
            {
                Buffer.MemoryCopy( srcPtr.ToPointer(), dst.ToPointer(), byteCount, byteCount );
            }
        }
    }
}
