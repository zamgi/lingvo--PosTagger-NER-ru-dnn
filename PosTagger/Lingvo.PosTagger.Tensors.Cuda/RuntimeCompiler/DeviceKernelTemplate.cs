using System;
using System.Collections.Generic;
using System.Linq;

namespace Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler
{
    /// <summary>
    /// 
    /// </summary>
    public class DeviceKernelTemplate
    {
        private readonly string _TemplateCode;
        private readonly List<string> _RequiredHeaders;
        private readonly HashSet<string> _RequiredConfigArgs;
        private readonly Dictionary<KernelConfig, byte[]> _PtxCache;

        public DeviceKernelTemplate( string templateCode, params string[] requiredHeaders )
        {
            _TemplateCode       = templateCode;
            _RequiredHeaders    = new List<string>( requiredHeaders );
            _RequiredConfigArgs = new HashSet<string>();
            _PtxCache           = new Dictionary<KernelConfig, byte[]>( KernelConfig.EqualityComparer.Inst );
        }

        public byte[] PtxForConfig( CudaCompiler compiler, KernelConfig config )
        {
            if ( _PtxCache.TryGetValue( config, out byte[] cachedResult ) )
            {
                return (cachedResult);
            }

            if ( !_RequiredConfigArgs.All( config.ContainsKey ) )
            {
                throw (new InvalidOperationException( $"All config arguments must be provided. Required: '{string.Join( ", ", _RequiredConfigArgs )}'." ));
            }

            // Checking this ensures that there is only one config argument that can evaluate to the same code,
            // which ensures that the ptx cacheing does not generate unnecessary combinations. Also, a mismatch
            // occurring here probably indicates a bug somewhere else.
            if ( !config.Keys.All( _RequiredConfigArgs.Contains ) )
            {
                throw (new InvalidOperationException( $"Config provides some unnecessary arguments. Required: '{string.Join( ", ", _RequiredConfigArgs )}'." ));
            }

            //return new DeviceKernelCode(config.ApplyToTemplate(templateCode), requiredHeaders.ToArray());
            var finalCode = config.ApplyToTemplate( _TemplateCode );

            byte[] result = compiler.CompileToPtx( finalCode, _RequiredHeaders.ToArray() );
            _PtxCache.Add( config, result );
            return (result);
        }
    }
}
