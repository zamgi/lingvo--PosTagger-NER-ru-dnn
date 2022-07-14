namespace Lingvo.PosTagger.Tensors.Cuda.DeviceCode
{
    /// <summary>
    /// 
    /// </summary>
    public static class NNThreads
    {
        public const int NUM_THREADS = 1024;
        public static int NumBlocks( int n ) => (n + NUM_THREADS - 1) / NUM_THREADS;
    }
}
