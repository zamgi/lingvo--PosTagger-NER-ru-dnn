namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    [Precompile]
    public class ElementwiseTriKernels : CudaCode
    {
        public ElementwiseTriKernels() : base( GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math" ) { }

        private static string GetFullCode()
        {
            var pg = new PermutationGenerator();

            AppendTTFunc( pg, "sin", "sin" );
            AppendTTFunc( pg, "cos", "cos" );
            AppendTTFunc( pg, "tan", "tan" );
            AppendTTFunc( pg, "asin", "asin" );
            AppendTTFunc( pg, "acos", "acos" );
            AppendTTFunc( pg, "atan", "atan" );
            AppendTTFunc( pg, "sinh", "sinh" );
            AppendTTFunc( pg, "cosh", "cosh" );
            AppendTTFunc( pg, "tanh", "tanhf" );

            pg.AddApplyTTT( "atan2", "*a = atan2f(*b, *c);" );

            AppendTTTFunc( pg, "addtanh", "AddTanh" );
            AppendTTTTFunc( pg, "addtanh3", "AddTanh3" );
            AppendTTTTFunc( pg, "addtanhD", "AddTanhD" );
            AppendTTTFunc( pg, "tanhD", "TanhD" );

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
