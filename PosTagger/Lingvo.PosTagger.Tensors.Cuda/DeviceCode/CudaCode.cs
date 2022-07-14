using Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler;

namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    public abstract class CudaCode : IPrecompilable
    {
        private readonly string _Code;
        private readonly string[] _RequiredHeaders;
        private byte[] _Ptx;

        protected CudaCode( string code, params string[] requiredHeaders )
        {
            _Code = code;
            _RequiredHeaders = requiredHeaders;
        }

        public byte[] GetPtx( CudaCompiler compiler )
        {
            if ( _Ptx == null )
            {
                Precompile( compiler );
            }
            return (_Ptx);
        }
        public void Precompile( CudaCompiler compiler ) => _Ptx = compiler.CompileToPtx( _Code, _RequiredHeaders );
    }

}
