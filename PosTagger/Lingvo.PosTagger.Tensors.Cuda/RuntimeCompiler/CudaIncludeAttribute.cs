using System;

namespace Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler
{
    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    public class CudaIncludeAttribute : Attribute
    {
        public CudaIncludeAttribute( string fieldName, string includeName )
        {
            FieldName   = fieldName;
            IncludeName = includeName;
        }
        public string FieldName   { get; }
        public string IncludeName { get; }
    }
}
