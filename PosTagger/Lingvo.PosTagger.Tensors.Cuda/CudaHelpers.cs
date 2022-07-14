using System;
using System.Collections.Generic;
using System.Linq;

using ManagedCuda.BasicTypes;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    public static class CudaHelpers
    {
        public static CUdeviceptr GetBufferStart(Tensor tensor) => ((CudaStorage)tensor.Storage).DevicePtrAtElement(tensor.StorageOffset);

        public static void ThrowIfDifferentDevices( params Tensor[] tensors )
        {
            IEnumerable<Tensor> nonNull = tensors.Where( x => x != null );
            if ( !nonNull.Any() )
            {
                return;
            }

            int device = GetDeviceId( nonNull.First() );

            if ( nonNull.Any( x => GetDeviceId( x ) != device ) )
            {
                throw (new InvalidOperationException( "All tensors must reside on the same device" ));
            }
        }

        public static int GetDeviceId( Tensor tensor ) => ((CudaStorage) tensor.Storage).DeviceId;
        public static TSCudaContext TSContextForTensor( Tensor tensor ) => ((CudaStorage) tensor.Storage).TSContext;
    }
}
