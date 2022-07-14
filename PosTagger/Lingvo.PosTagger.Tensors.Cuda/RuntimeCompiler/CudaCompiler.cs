using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

using Lingvo.PosTagger.Utils;

using ManagedCuda.NVRTC;

namespace Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler
{
    /// <summary>
    /// 
    /// </summary>

    public class CudaCompiler
    {
        private readonly Dictionary<string, string> _Includes;
        private readonly KernelDiskCache _DiskCache;
        private readonly string[] _Options;

        public CudaCompiler( KernelDiskCache diskCache, string[] options = null )
        {
            _Includes = new Dictionary<string, string>();
            _DiskCache = diskCache;
            _Options   = options;
            RegisterAttributeHeaders( Assembly.GetExecutingAssembly() );
        }

        public byte[] CompileToPtx( string code, params string[] prependIncludes )
        {
            // We manually prepend include files here, so that the header content forms part of the hash of the source
            // code. This means that changes to headers will correctly trigger a recompile.
            var finalCode = new StringBuilder();
            foreach ( var includeName in prependIncludes )
            {
                finalCode.Append( _Includes[ includeName ] ).Append( '\n' );
            }
            finalCode.Append( code );
            var finalCodeString = finalCode.ToString();

            return _DiskCache.Get( finalCodeString, DoCompile );
        }

        private byte[] DoCompile( string fullSource )
        {
            var rtc = new CudaRuntimeCompiler( fullSource, null );
            try
            {
                if ( _Options == null || _Options.Length == 0 )
                {
                    rtc.Compile( new string[] { } );
                }
                else
                {
                    Logger.WriteLine( $"Compiler Options: {string.Join( " ", _Options )}" );
                    rtc.Compile( _Options );
                    //rtc.Compile(new string[] { "--use_fast_math", "--gpu-architecture=compute_60" });
                }
            }
            catch
            {
                throw (new ApplicationException( $"Error compiling CUDA code: '{rtc.GetLogAsString()}'" ));
            }

            return (rtc.GetPTX());
        }
        private void RegisterAttributeHeaders( Assembly assembly )
        {
            foreach ( var t in assembly.TypesWithAttribute< CudaIncludeAttribute >( false ) )
            {
                foreach ( CudaIncludeAttribute attribute in t.attrs )
                {
                    var (name, content) = HeaderInfoFromAttribute( t.type, attribute );
                    _Includes.Add( name, content );
                }
            }
        }

        private static (string attrIncludeName, string content) HeaderInfoFromAttribute( Type containingType, CudaIncludeAttribute attribute )
        {
            var field   = containingType.GetField( attribute.FieldName, BindingFlags.Public | BindingFlags.Static );
            var content = (string) field.GetValue( null );
            return (attribute.IncludeName, content);
        }
    }
}
