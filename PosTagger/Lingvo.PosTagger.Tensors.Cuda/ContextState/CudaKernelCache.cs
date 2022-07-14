using System;
using System.Collections.Generic;

using ManagedCuda;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Tensors.Cuda.ContextState
{
    /// <summary>
    /// 
    /// </summary>

    public class CudaKernelCache : IDisposable
    {
        private readonly object _Locker;
        private readonly Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel> _ActiveKernels;
        public CudaKernelCache()
        {
            _Locker = new object();
            _ActiveKernels = new Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel>();
        }
        public void Dispose()
        {
            lock ( _Locker )
            {
                foreach ( KeyValuePair<Tuple<CudaContext, byte[], string>, CudaKernel> p in _ActiveKernels )
                {
                    CudaContext ctx = p.Key.Item1;
                    CudaKernel kernel = p.Value;

                    ctx.UnloadKernel( kernel );
                }
            }
        }
        public CudaKernel Get( CudaContext context, byte[] ptx, string kernelName )
        {
            lock ( _Locker )
            {
                try
                {
                    var t = Tuple.Create( context, ptx, kernelName );
                    if ( !_ActiveKernels.TryGetValue( t, out CudaKernel value ) )
                    {
                        value = context.LoadKernelPTX( ptx, kernelName );
                        _ActiveKernels.Add( t, value );                        
                    }
                    return (value);
                }
                catch ( Exception ex )
                {
                    Logger.WriteErrorLine( $"Exception: '{ex.Message}', Call stack: '{ex.StackTrace}'" );
                    throw;
                }
            }
        }
    }

}
