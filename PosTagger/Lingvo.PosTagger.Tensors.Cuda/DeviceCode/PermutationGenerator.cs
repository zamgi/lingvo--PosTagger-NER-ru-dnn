using System.Text;

namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    public class PermutationGenerator
    {
        public readonly StringBuilder _Buf;
        public PermutationGenerator() => _Buf = new StringBuilder( 0x1000 );
        public override string ToString() => _Buf.ToString();

        public void AddApplyT( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 1 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* v) const {{ {operatorCode} }} }};" );
                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> src, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );

                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, src);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName}()(&src.data[aOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTT( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 2 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();


                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};" );
                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );

                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);" );
                _Buf.AppendLine( $"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTTT( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 3 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();
                var dimsC = spec.TensorDims[ 2 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c) const {{ {operatorCode} }} }};" );
                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );

                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);" );
                _Buf.AppendLine( $"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);" );
                _Buf.AppendLine( $"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTTTT( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 4 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();
                var dimsC = spec.TensorDims[ 2 ].ToString();
                var dimsD = spec.TensorDims[ 3 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c, float *d) const {{ {operatorCode} }} }};" );
                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, TensorInfo<{indexType}> tensorD, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );

                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);" );
                _Buf.AppendLine( $"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);" );
                _Buf.AppendLine( $"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);" );
                _Buf.AppendLine( $"         const {indexType} dOffset = IndexToOffset < {indexType}, {dimsD}>::get(linearIndex, tensorD);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset], &tensorD.data[dOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTTTTT( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 5 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();
                var dimsC = spec.TensorDims[ 2 ].ToString();
                var dimsD = spec.TensorDims[ 3 ].ToString();
                var dimsE = spec.TensorDims[ 4 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c, float *d, float *e) const {{ {operatorCode} }} }};" );
                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, TensorInfo<{indexType}> tensorD, TensorInfo<{indexType}> tensorE, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );

                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);" );
                _Buf.AppendLine( $"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);" );
                _Buf.AppendLine( $"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);" );
                _Buf.AppendLine( $"         const {indexType} dOffset = IndexToOffset < {indexType}, {dimsD}>::get(linearIndex, tensorD);" );
                _Buf.AppendLine( $"         const {indexType} eOffset = IndexToOffset < {indexType}, {dimsE}>::get(linearIndex, tensorE);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset], &tensorD.data[dOffset], &tensorE.data[eOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }
        public void AddApplyTS( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 1 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{" );
                _Buf.AppendLine( "float b;" );
                _Buf.AppendLine( $"__device__ ConcreteOp_{kernelName}(float bVal) {{ this->b = bVal; }}" );
                _Buf.AppendLine( $"__device__ __forceinline__ void operator()(float* a) const {{ {operatorCode} }}" );
                _Buf.AppendLine( "};" );

                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );

                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, a);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b);" );
                _Buf.AppendLine( $"         op(&a.data[aOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTSS( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 1 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{" );
                _Buf.AppendLine( "float b;" );
                _Buf.AppendLine( "float c;" );
                _Buf.AppendLine( $"__device__ ConcreteOp_{kernelName}(float bVal, float cVal) {{ this->b = bVal; this->c = cVal; }}" );
                _Buf.AppendLine( $"__device__ __forceinline__ void operator()(float* a) const {{ {operatorCode} }}" );
                _Buf.AppendLine( "};" );

                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, float c, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );
                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, a);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b, c);" );
                _Buf.AppendLine( $"         op(&a.data[aOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTTS( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 2 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{" );
                _Buf.AppendLine( "float c;" );
                _Buf.AppendLine( $"__device__ ConcreteOp_{kernelName}(float cVal) {{ this->c = cVal; }}" );
                _Buf.AppendLine( $"__device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};" );

                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, float c, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );
                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);" );
                _Buf.AppendLine( $"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(c);" );
                _Buf.AppendLine( $"         op(&tensorA.data[aOffset], &tensorB.data[bOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTTSS( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 2 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{" );
                _Buf.AppendLine( "float c;" );
                _Buf.AppendLine( "float d;" );
                _Buf.AppendLine( $"__device__ ConcreteOp_{kernelName}(float cVal, float dVal) {{ this->c = cVal; this->d = dVal; }}" );
                _Buf.AppendLine( $"__device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};" );

                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, float c, float d, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );
                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);" );
                _Buf.AppendLine( $"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(c, d);" );
                _Buf.AppendLine( $"         op(&tensorA.data[aOffset], &tensorB.data[bOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddApplyTTTS( string kernelBaseName, string operatorCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 3 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();
                var dimsC = spec.TensorDims[ 2 ].ToString();

                _Buf.AppendLine( $"struct ConcreteOp_{kernelName} {{" );
                _Buf.AppendLine( "float d;" );
                _Buf.AppendLine( $"__device__ ConcreteOp_{kernelName}(float dVal) {{ this->d = dVal; }}" );
                _Buf.AppendLine( $"__device__ __forceinline__ void operator()(float* a, float *b, float *c) const {{ {operatorCode} }} }};" );

                _Buf.AppendLine( "extern \"C\" {" );
                _Buf.AppendLine( $"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, float d, __int64 totalElements)" );
                _Buf.AppendLine( "   {" );
                _Buf.AppendLine( $"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)" );
                _Buf.AppendLine( "      {" );
                _Buf.AppendLine( $"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);" );
                _Buf.AppendLine( $"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);" );
                _Buf.AppendLine( $"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);" );
                _Buf.AppendLine( $"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(d);" );
                _Buf.AppendLine( $"         op(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset]);" );
                _Buf.AppendLine( "      }" );
                _Buf.AppendLine( "   }" );
                _Buf.AppendLine( "}" );
            }
        }

        public void AddReduce( string kernelBaseName, string modifyOpCode, string reduceOpCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 2 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();
                _Buf.AppendFormat( "REDUCE_KERNELS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, kernelName, modifyOpCode, reduceOpCode );
            }
        }

        public void AddReduceNorm( string kernelBaseName )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 2 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                var dimsB = spec.TensorDims[ 1 ].ToString();
                _Buf.AppendFormat( "REDUCE_NORM_KERNELS({0}, {1}, {2}, {3})\n", indexType, dimsA, dimsB, kernelName );
            }
        }

        public void AddReduceAll( string kernelBaseName, string modifyOpCode, string reduceOpCode )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 1 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                _Buf.AppendFormat( "REDUCE_ALL_KERNELS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, kernelName, modifyOpCode, reduceOpCode );
            }
        }

        public void AddReduceAllNorm( string kernelBaseName )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 1 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                _Buf.AppendFormat( "REDUCE_ALL_NORM_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName );
            }
        }

        public void AddReduceAllSubSquare( string kernelBaseName )
        {
            foreach ( var spec in ApplySpecialization.AllSpecializations( 1 ) )
            {
                var kernelName = GetMangledName( kernelBaseName, spec );
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[ 0 ].ToString();
                _Buf.AppendFormat( "REDUCE_ALL_SUB_SQUARE_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName );
            }
        }

        public static string GetMangledName( string baseName, ApplySpecialization spec )
        {
            var sb = new StringBuilder( 20 * spec.TensorDims.Length );
            sb.Append( baseName );
            sb.Append( spec.Use32BitIndices ? "__int32" : "__int64" );
            foreach ( int dimSize in spec.TensorDims )
            {
                sb.Append( '_' ).Append( dimSize.ToString().Replace( '-', 'M' ) );
            }
            return (sb.ToString());
        }
    }
}
