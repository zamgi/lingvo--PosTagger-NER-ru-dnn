using System;
using System.Collections.Generic;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class WeightTensorFactory : IDisposable
    {
        private readonly List< WeightTensor > _Weights;
        public WeightTensorFactory() => _Weights = new List< WeightTensor >();
        public void Dispose()
        {
            foreach ( WeightTensor wt in _Weights )
            {
                wt.Dispose();
            }
            _Weights.Clear();
        }

        public WeightTensor CreateWeightTensor( int row, int column, int deviceId, bool cleanWeights = false, string name = "", bool isTrainable = false, ComputeGraphTensor graphToBind = null, NormType normType = NormType.None, bool needGradient = true )
        {
            var wt = new WeightTensor( new long[ 2 ] { row, column }, deviceId, name: name, isTrainable: isTrainable, normType: normType, graphToBind: graphToBind, needGradient: needGradient );
            if ( cleanWeights )
            {
                wt.CleanWeight();
            }
            _Weights.Add( wt );
            return wt;
        }
        public WeightTensor CreateWeightTensor( long[] sizes, int deviceId, bool cleanWeights = false, string name = "", ComputeGraphTensor graphToBind = null, NormType normType = NormType.None, bool needGradient = true )
        {
            var wt = new WeightTensor( sizes, deviceId, name, normType: normType, graphToBind: graphToBind, needGradient: needGradient );
            if ( cleanWeights )
            {
                wt.CleanWeight();
            }
            _Weights.Add( wt );
            return wt;
        }        
    }
}
