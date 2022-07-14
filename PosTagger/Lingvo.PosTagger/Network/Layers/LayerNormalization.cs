using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class LayerNormalization
    {
        private readonly WeightTensor _Alpha;
        private readonly WeightTensor _Beta;

        public LayerNormalization( string name, int dim, int deviceId, bool isTrainable, float learningRateFactor = 1.0f )
        {
            _Alpha = new WeightTensor( new long[ 2 ] { 1, dim }, 1.0f, deviceId, name: $"{name}.m_alpha", isTrainable, learningRateFactor );
            _Beta  = new WeightTensor( new long[ 2 ] { 1, dim },    0, deviceId, name: $"{name}.m_beta" , isTrainable, learningRateFactor );
        }

        public WeightTensor Norm( WeightTensor input, ComputeGraphTensor g ) => g.LayerNorm( input, _Alpha, _Beta, 1e-06f );

        public List< WeightTensor > GetParams() => new List< WeightTensor > { _Alpha, _Beta };
        public void Save( Model model )
        {
            _Alpha.Save( model );
            _Beta.Save( model );
        }
        public void Load( Model model )
        {
            _Alpha.Load( model );
            _Beta.Load( model );
        }
    }
}
