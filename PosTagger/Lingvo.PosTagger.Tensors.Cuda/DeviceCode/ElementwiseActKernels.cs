namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    [Precompile]
    public class ElementwiseActKernels : CudaCode
    {
        public ElementwiseActKernels() : base( GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math" ) { }

        private static string GetFullCode()
        {
            var pg = new PermutationGenerator();

            AppendTTFunc( pg, "sigmoid", "Sigmoid" );
            AppendTTTTFunc( pg, "addsigmoidD", "AddSigmoidD" );
            AppendTTTFunc( pg, "sigmoidD", "SigmoidD" );

            AppendTTFunc( pg, "relu", "relu" );
            AppendTTTFunc( pg, "relud", "relud" );
            AppendTTTTFunc( pg, "addrelud", "addrelud" );

            return pg.ToString();
        }

        private static void AppendTTFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyT( "t1_" + kernelBaseName, string.Format( "*v = {0}(*v);", func ) );
            pg.AddApplyTT( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b);", func ) );
        }
        private static void AppendTTTFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyTT( "t1_" + kernelBaseName, string.Format( "*a = {0}(*a, *b);", func ) );
            pg.AddApplyTTT( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b, *c);", func ) );
        }
        private static void AppendTTTTFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyTTT( "t1_" + kernelBaseName, string.Format( "*a = {0}(*a, *b, *c);", func ) );
            pg.AddApplyTTTT( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b, *c, *d);", func ) );
        }
    }
}
