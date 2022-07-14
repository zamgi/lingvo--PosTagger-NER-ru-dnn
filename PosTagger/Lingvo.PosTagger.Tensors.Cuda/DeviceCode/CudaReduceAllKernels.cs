namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    [Precompile]
    public class CudaReduceAllKernels : CudaCode
    {
        public CudaReduceAllKernels() : base( GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "ReduceAll", "ReduceAllMacros", "Math" ) { }

        private static string GetFullCode()
        {
            const string IDENTITY = "return a;";

            var pg = new PermutationGenerator();
            pg.AddReduceAll( "sumAll", IDENTITY, "return a + b;" );
            pg.AddReduceAll( "prodAll", IDENTITY, "return a * b;" );
            pg.AddReduceAll( "minAll", IDENTITY, "return min(a, b);" );
            pg.AddReduceAll( "maxAll", IDENTITY, "return max(a, b);" );

            pg.AddReduceAll( "e0_normAll", "return a != 0 ? 1 : 0;", "return a + b;" );
            pg.AddReduceAll( "e1_normAll", "return fabsf(a);", "return a + b;" );
            pg.AddReduceAll( "e2_normAll", "return a * a;", "return a + b;" );
            pg.AddReduceAllNorm( "en_normAll" );

            pg.AddReduceAllSubSquare( "subSquare" );

            return pg.ToString();
        }
    }
}
