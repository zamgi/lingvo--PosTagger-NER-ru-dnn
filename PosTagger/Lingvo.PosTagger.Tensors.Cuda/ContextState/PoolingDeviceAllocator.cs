using System;
using System.Collections.Generic;

using ManagedCuda;
using ManagedCuda.BasicTypes;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Tensors.Cuda.ContextState
{
    /// <summary>
    /// 
    /// </summary>
    public class PoolingDeviceAllocator : IDeviceAllocator
    {
        private const long MEMORY_ALIGNMENT = 256;

        private readonly CudaContext _Context;
        private readonly object _Locker = new object();

        private readonly ulong _AvailMemByteInTotal;
        private CUdeviceptr _MemPoolPtr;
        private readonly SizeT _StartMemAddr;
        private readonly SizeT _EndMemAddr;

        private readonly SortedDictionary< ulong /*addr*/, ulong /*size*/ > _UsedAddr2Size;

        public PoolingDeviceAllocator( CudaContext context, float memoryUsageRatio = 0.9f )
        {
            _Context = context;
            context.SetCurrent();

            _AvailMemByteInTotal = (ulong) ((ulong) context.GetFreeDeviceMemorySize() * memoryUsageRatio);

            _MemPoolPtr = context.AllocateMemory( _AvailMemByteInTotal );

            _StartMemAddr = _MemPoolPtr.Pointer;
            _EndMemAddr   = _StartMemAddr + _AvailMemByteInTotal;

            _UsedAddr2Size = new SortedDictionary< ulong, ulong >();

            Logger.WriteLine( $"Allocated Cuda memory: {_AvailMemByteInTotal}, address from '{_StartMemAddr}' to '{_EndMemAddr}'." );
        }
        public void Dispose()
        {
            _Context.SetCurrent();
            _Context.FreeMemory( _MemPoolPtr );
        }

        public float GetAllocatedMemoryRatio()
        {
            lock ( _Locker )
            {
                ulong allocatedMemByte = 0;
                foreach ( var size in _UsedAddr2Size.Values )
                {
                    allocatedMemByte += size;
                }

                return (float) ((float) allocatedMemByte / (float) _AvailMemByteInTotal);
            }
        }

        private CUdeviceptr AllocateMemory( ulong size )
        {
            lock ( _Locker )
            {
                SizeT currMemAddr = _StartMemAddr;
                SizeT currMemAddrEnd;

                foreach ( var (addr, sz) in _UsedAddr2Size )
                {
                    currMemAddrEnd = currMemAddr + size;

                    if ( _EndMemAddr < currMemAddrEnd )
                    {
                        GC.Collect(); // Collect unused tensor objects and free GPU memory
                        throw (new OutOfMemoryException( $"Out of GPU memory. Current memory usage = '{GetAllocatedMemoryRatio() * 100.0f:F}%'." ));
                    }

                    if ( currMemAddrEnd < addr )
                    {
                        _UsedAddr2Size.Add( currMemAddr, size );
                        return new CUdeviceptr( currMemAddr );
                    }
                    else
                    {
                        currMemAddr = addr + sz;
                    }
                }

                currMemAddrEnd = currMemAddr + size;
                if ( _EndMemAddr < currMemAddrEnd )
                {
                    GC.Collect(); // Collect unused tensor objects and free GPU memory
                    throw (new OutOfMemoryException( $"Out of GPU memory. Current memory usage = '{GetAllocatedMemoryRatio() * 100.0f:F}%'." ));
                }

                _UsedAddr2Size.Add( currMemAddr, size );
                return (new CUdeviceptr( currMemAddr ));
            }
        }

        public IDeviceMemory Allocate( long byteCount )
        {
            ulong size = PadToAlignment( byteCount, MEMORY_ALIGNMENT );

            lock ( _Locker )
            {
                CUdeviceptr buffer = AllocateMemory( size );

                var devMemory = new BasicDeviceMemory( buffer, _this =>
                {
                     lock ( _Locker )
                     {
                         _UsedAddr2Size.Remove( _this.Pointer );
                     }
                 });

                return (devMemory);
            }
        }

        private static ulong PadToAlignment( long size, long alignment ) => (ulong) (((size + alignment - 1) / alignment) * alignment);
    }
}
