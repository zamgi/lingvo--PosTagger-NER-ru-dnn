using System;

using Lingvo.PosTagger.Tensors;
using Lingvo.PosTagger.Tensors.Cpu;
using Lingvo.PosTagger.Tensors.Cuda;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    internal static class TensorAllocator
    {
        private static IAllocator[] _Allocator;
        private static TSCudaContext _CudaContext;
        private static int[] _DeviceIds;
        private static ProcessorTypeEnums _ArchType;

        public static void InitDevices( ProcessorTypeEnums archType, int[] ids, float memoryUsageRatio = 0.9f, string[] compilerOptions = null )
        {
            _ArchType  = archType;
            _DeviceIds = ids;
            _Allocator = new IAllocator[ _DeviceIds.Length ];

            if ( _ArchType == ProcessorTypeEnums.GPU )
            {
                foreach ( int id in _DeviceIds )
                {
                    Logger.WriteLine( $"Initialize device '{id}'" );
                }

                _CudaContext = new TSCudaContext( _DeviceIds, memoryUsageRatio, compilerOptions );
                _CudaContext.Precompile( Console.Write );
                _CudaContext.CleanUnusedPTX();
            }
        }

        public static IAllocator Allocator( int deviceId )
        {
            var idx = GetDeviceIdIndex( deviceId );
            var allocator = _Allocator[ idx ];
            if ( allocator == null )
            {
                if ( _ArchType == ProcessorTypeEnums.GPU )
                {
                    allocator = _Allocator[ idx ] = new CudaAllocator( _CudaContext, deviceId );
                }
                else
                {
                    allocator = _Allocator[ idx ] = new CpuAllocator();
                }                
            }            
            return (allocator);
        }

        private static int GetDeviceIdIndex( int id )
        {
            for ( var i = _DeviceIds.Length - 1; 0 <= i ; i-- )
            {
                if ( _DeviceIds[ i ] == id )
                {
                    return (i);
                }
            }
            
            throw (new ArgumentException( $"Failed to get deviceId '{id}', deviceId List = '{string.Join( ", ", _DeviceIds )}'" ));
        }
    }
}
