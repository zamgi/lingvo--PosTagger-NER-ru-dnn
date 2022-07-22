namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal interface IFeedForwardLayer : INeuralUnit
    {
        WeightTensor Process( ComputeGraphTensor g, WeightTensor inputT, int batchSize, float alpha = 1.0f );
    }
}
