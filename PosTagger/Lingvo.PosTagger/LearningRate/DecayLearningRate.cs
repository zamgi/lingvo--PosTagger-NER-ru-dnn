using System;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class DecayLearningRate : ILearningRate
    {
        private readonly float _StartLearningRate = 0.001f;
        private readonly int   _WarmupSteps       = 8000;
        private int            _WeightsUpdateCount;

        public DecayLearningRate( float startLearningRate, int warmupSteps, int weightsUpdatesCount )
        {
            Logger.WriteLine( $"Creating decay learning rate. StartLearningRate = '{startLearningRate}', WarmupSteps = '{warmupSteps}', WeightsUpdatesCount = '{weightsUpdatesCount}'" );
            _StartLearningRate  = startLearningRate;
            _WarmupSteps        = warmupSteps;
            _WeightsUpdateCount = weightsUpdatesCount;
        }

        public float GetCurrentLearningRate()
        {
            _WeightsUpdateCount++;
            float learningRate = _StartLearningRate * (float) (Math.Min( Math.Pow( _WeightsUpdateCount, -0.5 ), Math.Pow( _WarmupSteps, -1.5 ) * _WeightsUpdateCount ) / Math.Pow( _WarmupSteps, -0.5 ));
            return (learningRate);
        }
    }
}
