using System;
using System.Reflection;

using Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public class PrecompileAttribute : Attribute
    {
        public PrecompileAttribute() { }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IPrecompilable
    {
        void Precompile( CudaCompiler compiler );
    }

    /// <summary>
    /// 
    /// </summary>
    public static class PrecompileHelper
    {
        public static void PrecompileAllFields( object instance, CudaCompiler compiler )
        {
            Type type = instance.GetType();

            foreach ( FieldInfo field in type.GetFields() )
            {
                if ( typeof(IPrecompilable).IsAssignableFrom( field.FieldType ) )
                {
                    var precompilableField = (IPrecompilable) field.GetValue( instance );
                    Console.WriteLine( $"Compiling field '{field.Name}'" );
                    precompilableField.Precompile( compiler );
                }
            }
        }
    }
}
