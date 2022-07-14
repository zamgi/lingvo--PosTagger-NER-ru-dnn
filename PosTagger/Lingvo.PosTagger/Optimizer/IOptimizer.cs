using System.Collections.Generic;

using Lingvo.PosTagger.Network;

namespace Lingvo.PosTagger.Optimizer
{
    /// <summary>
    /// 
    /// </summary>
    public interface IOptimizer
    {
        void UpdateWeights( List< WeightTensor > model, int batchSize, float step_size, float regc, int iter );
    }
}
