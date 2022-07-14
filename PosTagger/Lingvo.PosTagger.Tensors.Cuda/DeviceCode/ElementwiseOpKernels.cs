namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    [Precompile]
    public class ElementwiseOpKernels : CudaCode
    {
        public ElementwiseOpKernels() : base( GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math" ) { }

        private static string GetFullCode()
        {
            var pg = new PermutationGenerator();

            AppendTTSFunc( pg, "add", "add_op" );
            AppendTTSFunc( pg, "sub", "sub_op" );
            AppendTTSFunc( pg, "rsub", "rsub_op" );
            AppendTTSFunc( pg, "mul", "mul_op" );
            AppendTTSFunc( pg, "div", "div_op" );
            AppendTTSFunc( pg, "rdiv", "rdiv_op" );
            AppendTTSFunc( pg, "mod", "Mod_op" );

            AppendTTSFunc( pg, "gt", "gt_op" );
            AppendTTSFunc( pg, "lt", "lt_op" );
            AppendTTSFunc( pg, "ge", "gt_op" );
            AppendTTSFunc( pg, "le", "le_op" );
            AppendTTSFunc( pg, "eq", "eq_op" );
            AppendTTSFunc( pg, "ne", "ne_op" );

            AppendTTTFunc( pg, "cadd", "add_op" );
            AppendTTTFunc( pg, "csub", "sub_op" );
            AppendTTTFunc( pg, "cmul", "mul_op" );
            AppendTTTFunc( pg, "cdiv", "div_op" );
            AppendTTTFunc( pg, "cmod", "Mod_op" );

            AppendTTTFunc( pg, "cgt", "gt_op" );
            AppendTTTFunc( pg, "clt", "lt_op" );
            AppendTTTFunc( pg, "cge", "gt_op" );
            AppendTTTFunc( pg, "cle", "le_op" );
            AppendTTTFunc( pg, "ceq", "eq_op" );
            AppendTTTFunc( pg, "cne", "ne_op" );

            AppendAtomicAdd( pg, "atomicAdd" );

            return pg.ToString();
        }

        private static void AppendAtomicAdd( PermutationGenerator pg, string kernelBaseName ) => pg.AddApplyTT( "t1_" + kernelBaseName, "atomicAdd(a, *b);" );
        private static void AppendTTSFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyTS( "t1_" + kernelBaseName, string.Format( "*a = {0}(*a, b);", func ) );
            pg.AddApplyTTS( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b, c);", func ) );
        }
        private static void AppendTTTFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyTT( "t1_" + kernelBaseName, string.Format( "*a = {0}(*a, *b);", func ) );
            pg.AddApplyTTT( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b, *c);", func ) );
        }
    }
}
