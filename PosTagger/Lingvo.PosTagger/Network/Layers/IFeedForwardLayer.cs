namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal interface IFeedForwardLayer : INeuralUnit
    {
        WeightTensor Process( WeightTensor inputT, int batchSize, ComputeGraphTensor g, float alpha = 1.0f );
    }
}
