using System;
using System.IO;
using System.Linq;
using System.Reflection;

using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;

using Lingvo.PosTagger.Tensors.Cuda.ContextState;
using Lingvo.PosTagger.Tensors.Cuda.Util;
using Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    public struct ScratchSpace
    {
        public int         Size;
        public CUdeviceptr Buffer;
    }

    /// <summary>
    /// 
    /// </summary>

    public class TSCudaContext : IDisposable
    {
        public const int MAX_DIMS = 25;
        private const string CACHE_DIR = @"cuda_cache\general";

        private readonly DeviceState[] _Devices;
        private readonly bool[,] _P2PAccess;
        private readonly int[] _DeviceIds;

        private readonly KernelDiskCache _DiskCache;

        private readonly CudaCompiler _Compiler;
        private readonly CudaKernelCache _KernelCache = new CudaKernelCache();

        public TSCudaContext( int[] deviceIds, float memoryUsageRatio = 0.9f, string[] compilerOptions = null )
        {
            _DeviceIds = deviceIds;

            _Devices = new DeviceState[ deviceIds.Length ];
            for ( int i = 0; i < deviceIds.Length; i++ )
            {
                _Devices[ i ] = new DeviceState( deviceIds[ i ], memoryUsageRatio );
            }
            _P2PAccess = EnablePeerAccess( _Devices.Select( x => x.CudaContext ).ToArray(), _Devices[ 0 ].CudaContext );

            _DiskCache = new KernelDiskCache( Path.Combine( Environment.CurrentDirectory, CACHE_DIR ) );
            _Compiler  = new CudaCompiler( _DiskCache, compilerOptions );

            OpRegistry.RegisterAssembly( Assembly.GetExecutingAssembly() );
        }

        private int GetDeviceIdIndex( int id )
        {
            for ( int i = 0; i < _DeviceIds.Length; i++ )
            {
                if ( _DeviceIds[ i ] == id )
                {
                    return (i);
                }
            }
            return (-1);
        }

        public CudaCompiler Compiler => _Compiler;
        public CudaKernelCache KernelCache => _KernelCache;

        public void Dispose()
        {
            _KernelCache.Dispose();
            foreach ( DeviceState device in _Devices )
            {
                device.Dispose();
            }
        }

        public void Synchronize( int deviceId )
        {
            var idx = GetDeviceIdIndex( deviceId );
            _Devices[ idx ].CudaContext.Synchronize();
        }

        public void SynchronizeAll()
        {
            foreach ( DeviceState device in _Devices )
            {
                device.CudaContext.Synchronize();
            }
        }

        public CudaContext CudaContextForDevice( int deviceId )
        {
            var idx = GetDeviceIdIndex( deviceId );
            return _Devices[ idx ].CudaContext;
        }

        public IDeviceAllocator AllocatorForDevice( int deviceId )
        {
            var idx = GetDeviceIdIndex( deviceId );
            return _Devices[ idx ].MemoryAllocator;
        }

        public CudaContext CudaContextForTensor( Tensor tensor ) => CudaContextForDevice( CudaHelpers.GetDeviceId( tensor ) );
        public ScratchSpace ScratchSpaceForDevice( int deviceId )
        {
            var idx = GetDeviceIdIndex( deviceId );
            return _Devices[ idx ].ScratchSpace;
        }

        public PooledObject< CudaBlas > BlasForDevice( int deviceId )
        {
            var idx = GetDeviceIdIndex( deviceId );
            return _Devices[ idx ].BlasHandles.Get();
        }

        public PooledObject<CudaBlas> BlasForTensor( Tensor tensor ) => BlasForDevice( CudaHelpers.GetDeviceId( tensor ) );

        public bool CanAccessPeer( int srcDevice, int peerDevice )
        {
            int srcDeviceIdx = GetDeviceIdIndex( srcDevice );
            int peerDeviceIdx = GetDeviceIdIndex( peerDevice );
            return _P2PAccess[ srcDeviceIdx, peerDeviceIdx ];
        }

        public CudaDeviceProperties DeviceInfoForContext( CudaContext cudaContext )
        {
            int idx = GetDeviceIdIndex( cudaContext.DeviceId );
            return _Devices[ idx ].DeviceInfo;
        }

        // Returns a matrix of [i, j] values where [i, j] is true iff device i can access device j
        private static bool[,] EnablePeerAccess( CudaContext[] cudaContexts, CudaContext restoreCurrent )
        {
            var result = new bool[ cudaContexts.Length, cudaContexts.Length ];

            for ( int i = 0; i < cudaContexts.Length; ++i )
            {
                for ( int j = 0; j < cudaContexts.Length; ++j )
                {
                    if ( i == j )
                    {
                        result[ i, j ] = true;
                    }
                    else
                    {
                        result[ i, j ] = EnablePeers( cudaContexts[ i ], cudaContexts[ j ] );
                    }
                }
            }

            restoreCurrent.SetCurrent();
            return (result);
        }

        private static bool EnablePeers( CudaContext src, CudaContext target )
        {
            if ( !src.DeviceCanAccessPeer( target ) )
            {
                return (false);
            }

            src.SetCurrent();

            try
            {
                CudaContext.EnablePeerAccess( target );
                return (true);
            }
            catch
            {
                return (false);
            }
        }

        public void Precompile( Action<string> precompileProgressWriter )
        {
            var assembly = Assembly.GetExecutingAssembly();
            foreach ( var t in assembly.TypesWithAttribute< PrecompileAttribute >( true ).Where( x => !x.type.IsAbstract ) )
            {
                precompileProgressWriter( "Precompiling " + t.type.Name + "\n" );

                var instance = (IPrecompilable) Activator.CreateInstance( t.type );
                instance.Precompile( Compiler );
            }
        }
        public void CleanUnusedPTX() => _DiskCache.CleanUnused();
    }
}
