namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    [Precompile]
    public class FillCopyKernels : CudaCode
    {
        public FillCopyKernels() : base( GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply" ) { }
        private static string GetFullCode()
        {
            var pg = new PermutationGenerator();
            pg.AddApplyTS( "fill", "*a = b;" );
            pg.AddApplyTT( "copy", "*a = *b;" );
            return pg.ToString();
        }
    }

}
