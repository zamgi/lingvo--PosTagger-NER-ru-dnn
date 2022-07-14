namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    [Precompile]
    public class ElementwiseKernels : CudaCode
    {
        public ElementwiseKernels() : base( GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math" ) { }

        private static string GetFullCode()
        {
            var pg = new PermutationGenerator();
            AppendTTFunc( pg, "abs", "fabs" );
            AppendTTFunc( pg, "neg", "-" );
            AppendTTFunc( pg, "sign", "sgn" );

            AppendTTFunc( pg, "sqrt", "sqrtf" );
            AppendTTFunc( pg, "rsqrt", "rsqrtf" );

            AppendTTFunc( pg, "exp", "expf" );
            AppendTTFunc( pg, "log", "logf" );
            AppendTTFunc( pg, "log1p", "log1p" );
            AppendTTFunc( pg, "floor", "floor" );
            AppendTTFunc( pg, "ceil", "ceil" );
            AppendTTFunc( pg, "round", "round" );
            AppendTTFunc( pg, "trunc", "trunc" );
            AppendTTFunc( pg, "frac", "Frac" );

            AppendTTTTTFunc( pg, "mulmuladd", "MulMulAdd" );
            AppendTTTTFunc( pg, "addmul", "AddMul" );
            AppendTTTSFunc( pg, "addmulv", "AddMul" );

            AppendTTTTFunc( pg, "adddiv", "AddDiv" );

            AppendTTTSFunc( pg, "maskfill", "MaskFill" );

            pg.AddApplyTS( "t1_pow", "*a = powf(*a, b);" );
            pg.AddApplyTTS( "t2_pow", "*a = powf(*b, c);" );
            pg.AddApplyTS( "t1_tpow", "*a = powf(b, *a);" );
            pg.AddApplyTTS( "t2_tpow", "*a = powf(c, *b);" );

            pg.AddApplyTTTS( "lerp", "*a = Lerp(*b, *c, d);" );

            pg.AddApplyTSS( "t1_clamp", "*a = Clamp(*a, b, c);" );
            pg.AddApplyTTSS( "t2_clamp", "*a = Clamp(*b, c, d);" );

            return pg.ToString();
        }

        private static void AppendTTFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyT( "t1_" + kernelBaseName, string.Format( "*v = {0}(*v);", func ) );
            pg.AddApplyTT( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b);", func ) );
        }
        private static void AppendTTSFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyTS( "t1_" + kernelBaseName, string.Format( "*a = {0}(*a, b);", func ) );
            pg.AddApplyTTS( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b, c);", func ) );
        }
        private static void AppendTTTSFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyTTS( "t1_" + kernelBaseName, string.Format( "*a = {0}(*a, *b, c);", func ) );
            pg.AddApplyTTTS( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b, *c, d);", func ) );
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
        private static void AppendTTTTTFunc( PermutationGenerator pg, string kernelBaseName, string func )
        {
            pg.AddApplyTTTT( "t1_" + kernelBaseName, string.Format( "*a = {0}(*a, *b, *c, *d);", func ) );
            pg.AddApplyTTTTT( "t2_" + kernelBaseName, string.Format( "*a = {0}(*b, *c, *d, *e);", func ) );
        }
    }
}
