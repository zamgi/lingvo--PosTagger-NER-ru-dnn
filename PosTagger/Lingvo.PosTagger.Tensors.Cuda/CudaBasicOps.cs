using System;
using System.Linq;

using Lingvo.PosTagger.Tensors.Core;
using Lingvo.PosTagger.Tensors.Cuda.DeviceCode;
using Lingvo.PosTagger.Tensors.Cuda.KernelOps;
using Lingvo.PosTagger.Tensors.Cuda.MatrixMul;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Tensors.Cuda
{
    /// <summary>
    /// 
    /// </summary>
    [OpsClass]
    public class CudaBasicOps
    {
        private readonly CopyOps _CopyOps;

        private readonly ElementwiseKernels _ElementwiseKernels = new ElementwiseKernels();
        private readonly ElementwiseOpKernels _ElementwiseOpKernels = new ElementwiseOpKernels();
        private readonly ElementwiseTriKernels _ElementwiseTriKernels = new ElementwiseTriKernels();
        private readonly ElementwiseActKernels _ElementwiseActKernels = new ElementwiseActKernels();

        private readonly FillCopyKernels _FillCopyKernels = new FillCopyKernels();

        private readonly CudaReduceKernels _CudaReduceKernels = new CudaReduceKernels();
        private readonly CudaReduceAllKernels _CudaReduceAllKernels = new CudaReduceAllKernels();

        private readonly VarStdKernels _VarStdKernels = new VarStdKernels();
        private readonly ReduceDimIndexKernels _ReduceDimIndexKernels = new ReduceDimIndexKernels();

        private readonly AdvFuncKernels _AdvFuncKernels = new AdvFuncKernels();

        public CudaBasicOps() => _CopyOps = new CopyOps( _FillCopyKernels );
        public static Tensor Concat( Tensor result, int dimension, params Tensor[] inputs ) => TensorConcatenation.Concat( result, dimension, inputs );

        [RegisterOpArgCount( "copy" )]
        public void CopyGpu(
            [OpArgStorageType( typeof( CudaStorage ) )] Tensor result,
            [OpArgStorageType( typeof( CudaStorage ) )] Tensor src )
        {
            long totalElements = result.ElementCount();
            if ( totalElements != src.ElementCount() ) throw (new InvalidOperationException( "Tensors must have equal numbers of elements" ));

            if ( src.DimensionCount == 0 )
            {
                return;
            }

            _CopyOps.CopyGpu( result, src, totalElements );
        }

        [RegisterOpArgCount( "copy" )]
        public void CopyCpuToGpu(
            [OpArgStorageType( typeof( CudaStorage ) )] Tensor result,
            [OpArgStorageType( typeof( Cpu.CpuStorage ) )] Tensor src )
        {
            long totalElements = result.ElementCount();
            if ( totalElements != src.ElementCount() ) throw (new InvalidOperationException( "Tensors must have equal numbers of elements" ));

            if ( src.DimensionCount == 0 )
            {
                return;
            }

            _CopyOps.CopyCpuToGpu( result, src, totalElements );
        }

        [RegisterOpArgCount( "copy" )]
        public void CopyGpuToCpu(
            [OpArgStorageType( typeof( Cpu.CpuStorage ) )] Tensor result,
            [OpArgStorageType( typeof( CudaStorage ) )] Tensor src )
        {
            long totalElements = result.ElementCount();
            if ( totalElements != src.ElementCount() ) throw (new InvalidOperationException( "Tensors must have equal numbers of elements" ));

            if ( src.DimensionCount == 0 )
            {
                return;
            }

            _CopyOps.CopyGpuToCpu( result, src, totalElements );
        }


        [RegisterOpStorageType("fill", typeof(CudaStorage))]
        public void Fill( Tensor result, float value ) => FillOp.Invoke( _FillCopyKernels, result, value );


        [RegisterOpStorageType("dot", typeof(CudaStorage))]
        public static Tensor Dot( Tensor result, Tensor lhs, Tensor rhs )
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor( lhs );
            if ( lhs.DimensionCount == 1 && rhs.DimensionCount == 1 )
            {
                return CudaMatrixMulDot.Dot( context, result, lhs, rhs );
            }
            else if ( lhs.DimensionCount == 2 && rhs.DimensionCount == 1 )
            {
                return CudaMatrixMulMV.Mul_M_V( context, result, lhs, rhs );
            }
            else if ( lhs.DimensionCount == 2 && rhs.DimensionCount == 2 )
            {
                return CudaMatrixMulMM.Mul_M_M( context, result, lhs, rhs );
            }
            else
            {
                throw (new NotSupportedException( $"Multiplication of {lhs.DimensionCount}D with {rhs.DimensionCount}D tensor is not supported" ));
            }
        }

        [RegisterOpStorageType("addmm", typeof(CudaStorage))]
        public static Tensor Addmm( Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2 )
        {
            try
            {
                TSCudaContext context = CudaHelpers.TSContextForTensor( src );
                if ( src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType) )
                {
                    throw (new InvalidOperationException( "All tensors must have the same element type" ));
                }
                if ( result != null && !(result.Storage is CudaStorage) ) throw (new ArgumentException( "result must be a CUDA tensor", nameof(result) ));
                if ( !(m1.Storage is CudaStorage) ) throw (new ArgumentException( "m1 must be a CUDA tensor", nameof(m1) ));
                if ( !(m2.Storage is CudaStorage) ) throw (new ArgumentException( "m2 must be a CUDA tensor", nameof(m2) ));
                if ( src.DimensionCount != 2 ) throw (new ArgumentException( "src must be a matrix", nameof(src) ));
                if ( m1.DimensionCount != 2 ) throw (new ArgumentException( "m1 must be a matrix", nameof(m1) ));
                if ( m2.DimensionCount != 2 ) throw (new ArgumentException( "m2 must be a matrix", nameof(m2) ));
                if ( src.Sizes[ 0 ] != m1.Sizes[ 0 ] || src.Sizes[ 1 ] != m2.Sizes[ 1 ] || m1.Sizes[ 1 ] != m2.Sizes[ 0 ] )
                {
                    throw (new InvalidOperationException( $"Size mismatch, srcSize0 = {src.Sizes[ 0 ]}, m1Size0 = {m1.Sizes[ 0 ]}, srcSize1 = {src.Sizes[ 1 ]}, m2Size1 = {m2.Sizes[ 1 ]}, m1Size1 = '{m1.Sizes[ 1 ]}', m2Size0 = '{m2.Sizes[ 0 ]}'" ));
                }

                Tensor writeTarget = TensorResultBuilder.GetWriteTarget( result, src, false, src.Sizes );

                if ( writeTarget != src )
                {
                    Ops.Copy( writeTarget, src );
                }

                CudaMatrixMulMM.Gemm( context, alpha, m1, m2, beta, writeTarget );

                return (writeTarget);
            }
            catch ( Exception ex )
            {
                Logger.WriteLine( $"Exception in Addmm: '{ex.Message}', Call stack: '{ex.StackTrace}'" );
                throw;
            }
        }

        [RegisterOpStorageType("addmmbatch", typeof(CudaStorage))]
        public static Tensor AddmmBatch( Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2 )
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor( src );
            if ( src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType) )
            {
                throw (new InvalidOperationException( "All tensors must have the same element type" ));
            }
            if ( result != null && !(result.Storage is CudaStorage) ) throw (new ArgumentException( "result must be a CUDA tensor", nameof(result) ));
            if ( !(m1.Storage is CudaStorage) ) throw (new ArgumentException( "m1 must be a CUDA tensor", nameof(m1) ));
            if ( !(m2.Storage is CudaStorage) ) throw (new ArgumentException( "m2 must be a CUDA tensor", nameof(m2) ));
            if ( src.DimensionCount != 3 ) throw (new ArgumentException( "src must be a matrix", nameof(src) ));
            if ( m1.DimensionCount != 3 ) throw (new ArgumentException( "m1 must be a matrix", nameof(m1) ));
            if ( m2.DimensionCount != 3 ) throw (new ArgumentException( "m2 must be a matrix", nameof(m2) ));
            if ( src.Sizes[ 1 ] != m1.Sizes[ 1 ] || src.Sizes[ 2 ] != m2.Sizes[ 2 ] || m1.Sizes[ 2 ] != m2.Sizes[ 1 ] )
            {
                throw (new InvalidOperationException( $"Size mismatch, srcSize0 = {src.Sizes[ 0 ]}, m1Size0 = {m1.Sizes[ 0 ]}, srcSize1 = {src.Sizes[ 1 ]}, m2Size1 = {m2.Sizes[ 1 ]}, m1Size1 = '{m1.Sizes[ 1 ]}', m2Size0 = '{m2.Sizes[ 0 ]}'" ));
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget( result, src, true, src.Sizes );

            if ( writeTarget != src )
            {
                Ops.Copy( writeTarget, src );
            }

            CudaMatrixMulMM.GemmBatch( context, alpha, m1, m2, beta, writeTarget );

            return (writeTarget);
        }

        [RegisterOpStorageType("abs", typeof(CudaStorage))]
        public Tensor Abs( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "abs", result, src );
        [RegisterOpStorageType("neg", typeof(CudaStorage))]
        public Tensor Neg( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "neg", result, src );
        [RegisterOpStorageType("sign", typeof(CudaStorage))]
        public Tensor Sign( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "sign", result, src );

        [RegisterOpStorageType("sqrt", typeof(CudaStorage))]
        public Tensor Sqrt( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "sqrt", result, src );

        [RegisterOpStorageType("rsqrt", typeof(CudaStorage))]
        public Tensor Rsqrt( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "rsqrt", result, src );

        [RegisterOpStorageType("exp", typeof(CudaStorage))]
        public Tensor Exp( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "exp", result, src );
        [RegisterOpStorageType("log", typeof(CudaStorage))]
        public Tensor Log( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "log", result, src );
        [RegisterOpStorageType("log1p", typeof(CudaStorage))]
        public Tensor Log1p( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "log1p", result, src );
        [RegisterOpStorageType("floor", typeof(CudaStorage))]
        public Tensor Floor( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "floor", result, src );
        [RegisterOpStorageType("ceil", typeof(CudaStorage))]
        public Tensor Ceil( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "ceil", result, src );
        [RegisterOpStorageType("round", typeof(CudaStorage))]
        public Tensor Round( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "round", result, src );
        [RegisterOpStorageType("trunc", typeof(CudaStorage))]
        public Tensor Trunc( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "trunc", result, src );
        [RegisterOpStorageType("frac", typeof(CudaStorage))]
        public Tensor Frac( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseKernels, "frac", result, src );

        [RegisterOpStorageType("sin", typeof(CudaStorage))]
        public Tensor Sin( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "sin", result, src );
        [RegisterOpStorageType("cos", typeof(CudaStorage))]
        public Tensor Cos( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "cos", result, src );
        [RegisterOpStorageType("tan", typeof(CudaStorage))]
        public Tensor Tan( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "tan", result, src );

        [RegisterOpStorageType("asin", typeof(CudaStorage))]
        public Tensor Asin( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "asin", result, src );
        [RegisterOpStorageType("acos", typeof(CudaStorage))]
        public Tensor Acos( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "acos", result, src );
        [RegisterOpStorageType("atan", typeof(CudaStorage))]
        public Tensor Atan( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "atan", result, src );

        [RegisterOpStorageType("sinh", typeof(CudaStorage))]
        public Tensor Sinh( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "sinh", result, src );
        [RegisterOpStorageType("cosh", typeof(CudaStorage))]
        public Tensor Cosh( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "cosh", result, src );
        [RegisterOpStorageType("tanh", typeof(CudaStorage))]
        public Tensor Tanh( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseTriKernels, "tanh", result, src );

        [RegisterOpStorageType("addtanhD", typeof(CudaStorage))]
        public Tensor AddTanhD( Tensor result, Tensor t, Tensor resW, Tensor resG ) => ElementwiseTTTTOp.Invoke( _ElementwiseTriKernels, "addtanhD", result, t, resW, resG );

        [RegisterOpStorageType("tanhD", typeof(CudaStorage))]
        public Tensor TanhD( Tensor result, Tensor resW, Tensor resG ) => ElementwiseTTTOp.Invoke( _ElementwiseTriKernels, "tanhD", result, resW, resG );

        [RegisterOpStorageType("addtanh", typeof(CudaStorage))]
        public Tensor AddTanh( Tensor result, Tensor x, Tensor y ) => ElementwiseTTTOp.Invoke( _ElementwiseTriKernels, "addtanh", result, x, y );

        [RegisterOpStorageType("addtanh3", typeof(CudaStorage))]
        public Tensor AddTanh3( Tensor result, Tensor x, Tensor y, Tensor z ) => ElementwiseTTTTOp.Invoke( _ElementwiseTriKernels, "addtanh3", result, x, y, z );

        [RegisterOpStorageType("sigmoidD", typeof(CudaStorage))]
        public Tensor SigmoidD( Tensor result, Tensor resW, Tensor resG ) => ElementwiseTTTOp.Invoke( _ElementwiseActKernels, "sigmoidD", result, resW, resG );

        [RegisterOpStorageType("sigmoid", typeof(CudaStorage))]
        public Tensor Sigmoid( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseActKernels, "sigmoid", result, src );

        [RegisterOpStorageType("addsigmoidD", typeof(CudaStorage))]
        public Tensor AddSigmoidD( Tensor result, Tensor t, Tensor resW, Tensor resG ) => ElementwiseTTTTOp.Invoke( _ElementwiseActKernels, "addsigmoidD", result, t, resW, resG );

        [RegisterOpStorageType("relu", typeof(CudaStorage))]
        public Tensor Relu( Tensor result, Tensor src ) => ElementwiseTTOp.Invoke( _ElementwiseActKernels, "relu", result, src );

        [RegisterOpStorageType("relud", typeof(CudaStorage))]
        public Tensor ReluD( Tensor result, Tensor w, Tensor g ) => ElementwiseTTTOp.Invoke( _ElementwiseActKernels, "relud", result, w, g );

        [RegisterOpStorageType("addrelud", typeof(CudaStorage))]
        public Tensor AddReluD( Tensor result, Tensor t, Tensor w, Tensor g ) => ElementwiseTTTTOp.Invoke( _ElementwiseActKernels, "addrelud", result, t, w, g );

        [RegisterOpStorageType("mulmuladd", typeof(CudaStorage))]
        public Tensor MulMulAdd( Tensor result, Tensor x, Tensor y, Tensor z, Tensor w ) => ElementwiseTTTTTOp.Invoke( _ElementwiseKernels, "mulmuladd", result, x, y, z, w );

        [RegisterOpStorageType("addmul", typeof(CudaStorage))]
        public Tensor AddMul( Tensor result, Tensor x, Tensor y, Tensor z ) => ElementwiseTTTTOp.Invoke( _ElementwiseKernels, "addmul", result, x, y, z );
        [RegisterOpStorageType("addmulv", typeof(CudaStorage))]
        public Tensor AddMulV( Tensor result, Tensor x, Tensor y, float z ) => ElementwiseTTTSOp.Invoke( _ElementwiseKernels, "addmulv", result, x, y, z );

        [RegisterOpStorageType("adddiv", typeof(CudaStorage))]
        public Tensor AddDiv( Tensor result, Tensor x, Tensor y, Tensor z ) => ElementwiseTTTTOp.Invoke( _ElementwiseKernels, "adddiv", result, x, y, z );

        [RegisterOpStorageType("maskfill", typeof(CudaStorage))]
        public Tensor MaskFill( Tensor result, Tensor t, Tensor mask, float defValue ) => ElementwiseTTTSOp.Invoke( _ElementwiseKernels, "maskfill", result, t, mask, defValue );

        [RegisterOpStorageType("atan2", typeof(CudaStorage))]
        public Tensor Atan2( Tensor result, Tensor srcY, Tensor srcX ) => Atan2Op.Invoke( _ElementwiseTriKernels, result, srcY, srcX );
        [RegisterOpStorageType("pow", typeof(CudaStorage))]
        public Tensor Pow( Tensor result, Tensor src, float value ) => ElementwiseTTSOp.Invoke( _ElementwiseKernels, "pow", result, src, value );
        [RegisterOpStorageType("tpow", typeof(CudaStorage))]
        public Tensor Tpow( Tensor result, float value, Tensor src ) => ElementwiseTTSOp.Invoke( _ElementwiseKernels, "tpow", result, src, value );
        [RegisterOpStorageType("lerp", typeof(CudaStorage))]
        public Tensor Lerp( Tensor result, Tensor srcA, Tensor srcB, float weight ) => LerpOp.Invoke( _ElementwiseKernels, result, srcA, srcB, weight );
        [RegisterOpStorageType("clamp", typeof(CudaStorage))]
        public Tensor Clamp( Tensor result, Tensor src, float min, float max ) => ClampOp.Invoke( _ElementwiseKernels, result, src, min, max );

        [RegisterOpStorageType("addv", typeof(CudaStorage))]
        public Tensor Add( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "add", result, rhs, lhs );
        [RegisterOpStorageType("subv", typeof(CudaStorage))]
        public Tensor Sub( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "sub", result, rhs, lhs );
        [RegisterOpStorageType("rsubv", typeof(CudaStorage))]
        public Tensor Sub( Tensor result, float rhs, Tensor lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "rsub", result, lhs, rhs );
        [RegisterOpStorageType("mulv", typeof(CudaStorage))]
        public Tensor Mul( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "mul", result, rhs, lhs );
        [RegisterOpStorageType("divv", typeof(CudaStorage))]
        public Tensor Div( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "div", result, rhs, lhs );
        [RegisterOpStorageType("rdivv", typeof(CudaStorage))]
        public Tensor Div( Tensor result, float rhs, Tensor lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "rdiv", result, lhs, rhs );
        [RegisterOpStorageType("modv", typeof(CudaStorage))]
        public Tensor Mod( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "mod", result, rhs, lhs );

        [RegisterOpStorageType("gtValue", typeof(CudaStorage))]
        public Tensor GreaterThan( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "gt", result, rhs, lhs );
        [RegisterOpStorageType("ltValue", typeof(CudaStorage))]
        public Tensor LessThan( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "lt", result, rhs, lhs );
        [RegisterOpStorageType("geValue", typeof(CudaStorage))]
        public Tensor GreaterOrEqual( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "ge", result, rhs, lhs );
        [RegisterOpStorageType("leValue", typeof(CudaStorage))]
        public Tensor LessOrEqual( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "le", result, rhs, lhs );
        [RegisterOpStorageType("eqValue", typeof(CudaStorage))]
        public Tensor EqualTo( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "eq", result, rhs, lhs );
        [RegisterOpStorageType("neValue", typeof(CudaStorage))]
        public Tensor NotEqual( Tensor result, Tensor rhs, float lhs ) => ElementwiseTTSOp.Invoke( _ElementwiseOpKernels, "ne", result, rhs, lhs );


        [RegisterOpStorageType("addt", typeof(CudaStorage))]
        public Tensor Add( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cadd", result, rhs, lhs );


        [RegisterOpStorageType("atomicadd", typeof(CudaStorage))]
        public Tensor AtomicAdd( Tensor result, Tensor rhs ) => ElementwiseAtomicAddOp.Invoke( _ElementwiseOpKernels, result, rhs );


        [RegisterOpStorageType("subt", typeof(CudaStorage))]
        public Tensor Sub( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "csub", result, rhs, lhs );

        [RegisterOpStorageType("mult", typeof(CudaStorage))]
        public Tensor Mul( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cmul", result, rhs, lhs );

        [RegisterOpStorageType("divt", typeof(CudaStorage))]
        public Tensor Div( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cdiv", result, rhs, lhs );

        [RegisterOpStorageType("modt", typeof(CudaStorage))]
        public Tensor Mod( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cmod", result, rhs, lhs );

        [RegisterOpStorageType("gtTensor", typeof(CudaStorage))]
        public Tensor GreaterThan( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cgt", result, rhs, lhs );
        [RegisterOpStorageType("ltTensor", typeof(CudaStorage))]
        public Tensor LessThan( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "clt", result, rhs, lhs );
        [RegisterOpStorageType("geTensor", typeof(CudaStorage))]
        public Tensor GreaterOrEqual( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cge", result, rhs, lhs );
        [RegisterOpStorageType("leTensor", typeof(CudaStorage))]
        public Tensor LessOrEqual( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cle", result, rhs, lhs );
        [RegisterOpStorageType("eqTensor", typeof(CudaStorage))]
        public Tensor EqualTo( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "ceq", result, rhs, lhs );
        [RegisterOpStorageType("neTensor", typeof(CudaStorage))]
        public Tensor NotEqual( Tensor result, Tensor rhs, Tensor lhs ) => ElementwiseTTTOp.Invoke( _ElementwiseOpKernels, "cne", result, rhs, lhs );


        [RegisterOpStorageType("sum", typeof(CudaStorage))]
        public Tensor Sum( Tensor result, Tensor src, int dimension ) => ReductionOp.Invoke( _CudaReduceKernels, "sum", 0.0f, ReduceInitType.GivenValue, result, src, dimension );
        [RegisterOpStorageType("prod", typeof(CudaStorage))]
        public Tensor Prod( Tensor result, Tensor src, int dimension ) => ReductionOp.Invoke( _CudaReduceKernels, "prod", 1.0f, ReduceInitType.GivenValue, result, src, dimension );
        [RegisterOpStorageType("min", typeof(CudaStorage))]
        public Tensor Min( Tensor result, Tensor src, int dimension ) => ReductionOp.Invoke( _CudaReduceKernels, "min", 0.0f, ReduceInitType.MaxValue, result, src, dimension );
        [RegisterOpStorageType("max", typeof(CudaStorage))]
        public Tensor Max( Tensor result, Tensor src, int dimension ) => ReductionOp.Invoke( _CudaReduceKernels, "max", 0.0f, ReduceInitType.MinValue, result, src, dimension );

        [RegisterOpStorageType("argmin", typeof(CudaStorage))]
        public Tensor Argmin( Tensor result, Tensor src, int dimension ) => _ReduceDimIndexKernels.ArgMin( result, src, dimension );

        [RegisterOpStorageType("argmax", typeof(CudaStorage))]
        public Tensor Argmax( Tensor result, Tensor src, int dimension ) => _ReduceDimIndexKernels.ArgMax( result, src, dimension );


        [RegisterOpStorageType("mean", typeof(CudaStorage))]
        public Tensor Mean( Tensor result, Tensor src, int dimension )
        {
            long[] requiredOutputSize = src.Sizes.ToArray();
            requiredOutputSize[ dimension ] = 1;
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget( result, src, false, requiredOutputSize );

            Sum( writeTarget, src, dimension );
            Div( writeTarget, writeTarget, src.Sizes[ dimension ] );
            return (writeTarget);
        }

        [RegisterOpStorageType("norm", typeof(CudaStorage))]
        public Tensor Norm( Tensor result, Tensor src, int dimension, float value )
        {
            if ( value == 0 )
            {
                return ReductionOp.Invoke( _CudaReduceKernels, "e0_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension );
            }
            else if ( value == 1 )
            {
                return ReductionOp.Invoke( _CudaReduceKernels, "e1_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension );
            }
            else if ( value == 2 )
            {
                Tensor writeTarget = ReductionOp.Invoke( _CudaReduceKernels, "e2_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension );
                Pow( writeTarget, writeTarget, 0.5f );
                return (writeTarget);
            }
            else
            {
                Tensor writeTarget = ReductionOp.Invoke( _CudaReduceKernels, "en_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension, value );
                Pow( writeTarget, writeTarget, 1.0f / value );
                return (writeTarget);
            }
        }

        [RegisterOpStorageType("std", typeof(CudaStorage))]
        public Tensor Std( Tensor result, Tensor src, int dimension, bool normByN ) => _VarStdKernels.Std( result, src, dimension, normByN );
        [RegisterOpStorageType("var", typeof(CudaStorage))]
        public Tensor Var( Tensor result, Tensor src, int dimension, bool normByN ) => _VarStdKernels.Var( result, src, dimension, normByN );


        [RegisterOpStorageType("indexselect", typeof(CudaStorage))]
        public Tensor IndexSelect( Tensor result, Tensor src, Tensor indice ) => _AdvFuncKernels.IndexSelect( result, src, indice );

        [RegisterOpStorageType("indexselectgrad", typeof(CudaStorage))]
        public Tensor IndexSelectGrad( Tensor grad, Tensor adj, Tensor indice ) => _AdvFuncKernels.IndexSelectGrad( grad, adj, indice );


        [RegisterOpStorageType("buildsrctgtmask", typeof(CudaStorage))]
        public Tensor BuildSrcTgtMask( Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int srcPaddedSeqLength, int tgtPaddedSeqLength, float value, float maskedValue )
            => _AdvFuncKernels.BuildSrcTgtMask( result, srcOriginalLengths, tgtOriginalLengths, srcPaddedSeqLength, tgtPaddedSeqLength, value, maskedValue );


        [RegisterOpStorageType("buildselfmask", typeof(CudaStorage))]
        public Tensor BuildSelfMask( Tensor result, Tensor originalLengths, int paddedSeqLength, float value, float maskedValue )
            => _AdvFuncKernels.BuildSelfMask( result, originalLengths, paddedSeqLength, value, maskedValue );


        [RegisterOpStorageType("buildselftrimask", typeof(CudaStorage))]
        public Tensor BuildSelfTriMask( Tensor result, Tensor originalLengths, int paddedSeqLength, float value, float maskedValue )
            => _AdvFuncKernels.BuildSelfTriMask( result, originalLengths, paddedSeqLength, value, maskedValue );

        [RegisterOpStorageType("buildtrimask", typeof(CudaStorage))]
        public Tensor BuildTriMask( Tensor result, float value, float maskedValue ) => _AdvFuncKernels.BuildTriMask( result, value, maskedValue );

        [RegisterOpStorageType("softmax", typeof(CudaStorage))]
        public Tensor Softmax( Tensor result, Tensor src ) => _AdvFuncKernels.Softmax( result, src );

        [RegisterOpStorageType("softmaxgrad", typeof(CudaStorage))]
        public Tensor SoftmaxGrad( Tensor grad, Tensor adj, Tensor val, bool addGrad = true ) => _AdvFuncKernels.SoftmaxGrad( grad, adj, val, addGrad );

        [RegisterOpStorageType("layernorm", typeof(CudaStorage))]
        public Tensor LayerNorm( Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f ) => _AdvFuncKernels.LayerNorm( result, src, alpha, beta, eps );
        [RegisterOpStorageType("layernormgrad", typeof(CudaStorage))]
        public Tensor LayerNormGrad( Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-09f ) => _AdvFuncKernels.LayerNormGrad( outGrad, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps );


        [RegisterOpStorageType("addlayernorm", typeof(CudaStorage))]
        public Tensor AddLayerNorm( Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-09f ) => _AdvFuncKernels.AddLayerNorm( result, src1, src2, alpha, beta, eps );
        [RegisterOpStorageType("addlayernormgrad", typeof(CudaStorage))]
        public void AddLayerNormGrad( Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-09f ) => _AdvFuncKernels.AddLayerNormGrad( out1Grad, out2Grad, alphaGrad, betaGrad, inGrad, y, x1, x2, alpha, beta, eps );

        [RegisterOpStorageType("adam", typeof(CudaStorage))]
        public Tensor Adam( Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps )
            => _AdvFuncKernels.Adam( weight, gradient, v, m, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps );


        [RegisterOpStorageType("rmsprop", typeof(CudaStorage))]
        public Tensor RMSProp( Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps )
            => _AdvFuncKernels.RMSProp( weight, gradient, cache, batchSize, step_size, clipval, regc, decay_rate, eps );

        [RegisterOpStorageType("sumall", typeof(CudaStorage))]
        public Tensor SumAll( Tensor result, Tensor src ) => ReduceAllOp.Invoke( _CudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "sumAll", result, src );

        [RegisterOpStorageType("prodall", typeof(CudaStorage))]
        public Tensor ProdAll( Tensor result, Tensor src ) => ReduceAllOp.Invoke( _CudaReduceAllKernels, 1.0f, ReduceInitType.GivenValue, "prodAll", result, src );

        [RegisterOpStorageType("minall", typeof(CudaStorage))]
        public Tensor MinAll( Tensor result, Tensor src ) => ReduceAllOp.Invoke( _CudaReduceAllKernels, 0, ReduceInitType.MaxValue, "minAll", result, src );

        [RegisterOpStorageType("maxall", typeof(CudaStorage))]
        public Tensor MaxAll( Tensor result, Tensor src ) => ReduceAllOp.Invoke( _CudaReduceAllKernels, 0, ReduceInitType.MinValue, "maxAll", result, src );

        [RegisterOpStorageType("meanall", typeof(CudaStorage))]
        public Tensor MeanAll( Tensor result, Tensor src )
        {
            if ( src.DimensionCount == 0 || src.ElementCount() == 0 ) throw (new ArgumentException( "src must be a non-empty tensor" ));

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget( result, src, false, 1 );
            SumAll( writeTarget, src );
            Div( writeTarget, writeTarget, src.ElementCount() );
            return (writeTarget);
        }

        [RegisterOpStorageType("normall", typeof(CudaStorage))]
        public Tensor NormAll( Tensor result, Tensor src, float value )
        {
            if ( value == 0 )
            {
                return ReduceAllOp.Invoke( _CudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e0_normAll", result, src );
            }
            else if ( value == 1 )
            {
                return ReduceAllOp.Invoke( _CudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e1_normAll", result, src );
            }
            else if ( value == 2 )
            {

                Tensor writeTarget = ReduceAllOp.Invoke( _CudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e2_normAll", result, src );
                Pow( writeTarget, writeTarget, 0.5f );
                return (writeTarget);
            }
            else
            {
                Tensor writeTarget = ReduceAllOp.Invoke( _CudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_normAll", result, src, value );
                Pow( writeTarget, writeTarget, 1.0f / value );
                return (writeTarget);
            }
        }

        [RegisterOpStorageType("varall", typeof(CudaStorage))]
        public Tensor VarAll( Tensor result, Tensor src )
        {
            if ( src.DimensionCount == 0 || src.ElementCount() == 0 ) throw (new ArgumentException( "src must be a non-empty tensor" ));

            float mean = Ops.MeanAll( src );
            Tensor writeTarget = ReduceAllOp.Invoke( _CudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_norm", result, src, mean );
            Div( writeTarget, writeTarget, src.ElementCount() - 1 );
            return (writeTarget);
        }

        [RegisterOpStorageType("stdall", typeof(CudaStorage))]
        public Tensor StdAll( Tensor result, Tensor src )
        {
            Tensor writeTarget = VarAll( result, src );
            Pow( writeTarget, writeTarget, 0.5f );
            return (writeTarget);
        }
    }
}
