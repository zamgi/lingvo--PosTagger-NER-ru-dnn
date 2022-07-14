namespace Lingvo.PosTagger.Tensors.Cpu
{
    /// <summary>
    /// 
    /// </summary>
    public class ConvolutionDesc2d
    {
        public int kW;
        public int kH;
        public int dW;
        public int dH;
        public int padW;
        public int padH;

        public ConvolutionDesc2d( int kW, int kH, int dW, int dH, int padW, int padH )
        {
            this.kW = kW;
            this.kH = kH;
            this.dW = dW;
            this.dH = dH;
            this.padW = padW;
            this.padH = padH;
        }
    }
}
