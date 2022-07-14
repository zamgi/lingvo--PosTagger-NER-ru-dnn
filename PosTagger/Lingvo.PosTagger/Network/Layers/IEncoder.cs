namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal interface IEncoder : INeuralUnit
    {
        WeightTensor Encode( WeightTensor rawInput, int batchSize, ComputeGraphTensor g, WeightTensor srcSelfMask );
        void Reset( WeightTensorFactory weightFactory, int batchSize );
    }
}
