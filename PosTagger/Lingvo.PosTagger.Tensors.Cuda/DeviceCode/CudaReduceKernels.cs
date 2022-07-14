namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    [Precompile]
    public class CudaReduceKernels : CudaCode
    {
        public CudaReduceKernels() : base( GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "Math" ) { }

        private static string GetFullCode()
        {
            const string IDENTITY = "return a;";

            var pg = new PermutationGenerator();
            pg.AddReduce( "sum", IDENTITY, "return a + b;" );
            pg.AddReduce( "prod", IDENTITY, "return a * b;" );
            pg.AddReduce( "min", IDENTITY, "return min(a, b);" );
            pg.AddReduce( "max", IDENTITY, "return max(a, b);" );

            pg.AddReduce( "e0_norm", "return a != 0 ? 1 : 0;", "return a + b;" );
            pg.AddReduce( "e1_norm", "return fabsf(a);", "return a + b;" );
            pg.AddReduce( "e2_norm", "return a * a;", "return a + b;" );
            pg.AddReduceNorm( "en_norm" );

            return pg.ToString();
        }
    }
}
