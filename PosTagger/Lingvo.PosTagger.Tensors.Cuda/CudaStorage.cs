using System;

using ManagedCuda;
using ManagedCuda.BasicTypes;

using Lingvo.PosTagger.Tensors.Cuda.ContextState;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    public class CudaStorage : Storage
    {
        private readonly CudaContext _Context;
        private readonly CUdeviceptr _DeviceBuffer;
        private IDeviceMemory _BufferHandle;

        public CudaStorage( IAllocator allocator, TSCudaContext tsContext, CudaContext context, DType ElementType, long elementCount )
            : base( allocator, ElementType, elementCount )
        {
            TSContext = tsContext;
            _Context  = context;

            _BufferHandle = tsContext.AllocatorForDevice( DeviceId ).Allocate( ByteLength );
            _DeviceBuffer = _BufferHandle.Pointer;
        }
        protected override void Destroy()
        {
            if ( _BufferHandle != null )
            {
                _BufferHandle.Free();
                _BufferHandle = null;
            }
        }

        public TSCudaContext TSContext { get; private set; }
        public int DeviceId => _Context.DeviceId;

        public override string LocationDescription() => "CUDA:" + _Context.DeviceId;

        public CUdeviceptr DevicePtrAtElement( long index )
        {
            long offset = ElementType.Size() * index;
            return (new CUdeviceptr( _DeviceBuffer.Pointer + offset ));
        }

        public override int[] GetElementsAsInt( long index, int length )
        {
            CUdeviceptr ptr = DevicePtrAtElement( index );

            if ( ElementType == DType.Int32 ) 
            {
                var result = new int[ length ];
                _Context.CopyToHost( result, ptr );
                return (result);
            }
            else
            {
                throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
            }
        }
        public override void SetElementsAsInt( long index, int[] value )
        {
            CUdeviceptr ptr = DevicePtrAtElement( index );

            if ( ElementType == DType.Int32 ) 
            { 
                _Context.CopyToDevice( ptr, value );
            }
            else
            {
                throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
            }
        }

        public override float GetElementAsFloat( long index )
        {
            CUdeviceptr ptr = DevicePtrAtElement( index );
            try
            {
                if ( ElementType == DType.Float32 )
                { 
                    var result = new float[ 1 ]; 
                    _Context.CopyToHost( result, ptr ); 
                    return (result[ 0 ]);
                }
                else if ( ElementType == DType.Float64 ) 
                { 
                    var result = new double[ 1 ];
                    _Context.CopyToHost( result, ptr );
                    return ((float) result[ 0 ]); 
                }
                else if ( ElementType == DType.Int32 ) 
                { 
                    var result = new int[ 1 ]; 
                    _Context.CopyToHost( result, ptr ); 
                    return (result[ 0 ]);
                }
                else if ( ElementType == DType.UInt8 ) 
                {
                    var result = new byte[ 1 ]; 
                    _Context.CopyToHost( result, ptr ); 
                    return (result[ 0 ]);
                }
                else
                {
                    throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
                }
            }
            catch ( Exception ex )
            {
                Logger.WriteLine( $"Failed to get element as float from addr = '{ptr.Pointer}'" );
                Logger.WriteLine( $"Exception: {ex.Message}, Call stack: {ex.StackTrace}" );
                throw;
            }
        }
        public override float[] GetElementsAsFloat( long index, int length )
        {
            CUdeviceptr ptr = DevicePtrAtElement( index );

            if ( ElementType == DType.Float32 )
            {
                float[] result = new float[ length ];
                _Context.CopyToHost( result, ptr ); 
                return result; 
            }
            else
            {
                throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
            }
        }

        public override void SetElementAsFloat( long index, float value )
        {
            CUdeviceptr ptr = DevicePtrAtElement( index );

            if ( ElementType == DType.Float32 )
            {
                _Context.CopyToDevice( ptr, value );
            }
            else if ( ElementType == DType.Float64 ) 
            { 
                _Context.CopyToDevice( ptr, (double) value ); 
            }
            else if ( ElementType == DType.Int32 )
            { 
                _Context.CopyToDevice( ptr, (int) value ); 
            }
            else if ( ElementType == DType.UInt8 )
            { 
                _Context.CopyToDevice( ptr, (byte) value ); 
            }
            else
            {
                throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
            }
        }
        public override void SetElementsAsFloat( long index, float[] value )
        {
            _Context.SetCurrent();

            CUdeviceptr ptr = DevicePtrAtElement( index );

            if ( ElementType == DType.Float32 ) 
            { 
                _Context.CopyToDevice( ptr, value );
            }
            else
            {
                throw (new NotSupportedException( $"Element type '{ElementType}' not supported." ));
            }
        }

        public override void CopyToStorage( long storageIndex, IntPtr src, long byteCount )
        {
            CUdeviceptr dstPtr = DevicePtrAtElement( storageIndex );
            _Context.SetCurrent();
            _Context.CopyToDevice( dstPtr, src, byteCount );
        }
        public override void CopyFromStorage( IntPtr dst, long storageIndex, long byteCount )
        {
            CUdeviceptr srcPtr = DevicePtrAtElement( storageIndex );

            // Call this method directly instead of CudaContext.CopyToHost because this method supports a long byteCount
            // CopyToHost only supports uint byteCount.
            CUResult res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2( dst, srcPtr, byteCount );
            if ( res != CUResult.Success )
            {
                throw (new CudaException( res ));
            }
        }
    }
}
