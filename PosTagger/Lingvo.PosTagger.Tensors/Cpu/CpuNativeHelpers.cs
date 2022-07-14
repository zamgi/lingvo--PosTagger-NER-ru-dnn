using System;

namespace Lingvo.PosTagger.Tensors.Cpu
{
    /// <summary>
    /// 
    /// </summary>
    public static class CpuNativeHelpers
    {
        public static IntPtr GetBufferStart( Tensor tensor )
        {
            IntPtr buffer = ((CpuStorage) tensor.Storage)._Buffer;
            return PtrAdd( buffer, tensor.StorageOffset * tensor.ElementType.Size() );
        }

        private static IntPtr PtrAdd( IntPtr ptr, long offset ) => new IntPtr( ptr.ToInt64() + offset );
    }
}
