using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class PositionwiseFeedForward
    {
        private readonly LayerNormalization _LayerNorm2;
        private readonly FeedForwardLayer   _FeedForwardLayer1;
        private readonly FeedForwardLayer   _FeedForwardLayer2;
        private readonly string             _Name;
        private readonly float              _DropoutRatio;

        public PositionwiseFeedForward( string name, int hiddenDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f )
        {
            _Name         = name;
            _DropoutRatio = dropoutRatio;

            _LayerNorm2        = new LayerNormalization( $"{name}.layerNorm2", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor );
            _FeedForwardLayer1 = new FeedForwardLayer( $"{name}.feedForwardLayer1", hiddenDim, hiddenDim * 4, _DropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor );
            _FeedForwardLayer2 = new FeedForwardLayer( $"{name}.feedForwardLayer2", hiddenDim * 4, hiddenDim, _DropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor );
        }

        public WeightTensor Perform( WeightTensor input, int batchSize, ComputeGraphTensor graph )
        {
            using ComputeGraphTensor g = graph.CreateSubGraph( $"{_Name}_PositionwiseFeedForward" );
            var inputNorm = _LayerNorm2.Norm( input, g );

            //Feed forward
            WeightTensor ffnResult = _FeedForwardLayer1.Process( inputNorm, batchSize, g );
            WeightTensor reluFFNResult = g.Relu( ffnResult, inPlace: true );
            WeightTensor ffn2Result = _FeedForwardLayer2.Process( reluFFNResult, batchSize, g );

            //Skip connection and layer normaliztion
            WeightTensor addFFNResult = graph.Add( ffn2Result, input, inPlace: true );
            return (addFFNResult);
        }

        public List< WeightTensor > GetParams()
        {
            var response = new List< WeightTensor >();

            response.AddRange( _LayerNorm2.GetParams() );
            response.AddRange( _FeedForwardLayer1.GetParams() );
            response.AddRange( _FeedForwardLayer2.GetParams() );

            return (response);
        }
        public void Save( Model model )
        {
            _LayerNorm2.Save( model );
            _FeedForwardLayer1.Save( model );
            _FeedForwardLayer2.Save( model );
        }
        public void Load( Model model )
        {
            _LayerNorm2.Load( model );
            _FeedForwardLayer1.Load( model );
            _FeedForwardLayer2.Load( model );
        }
    }
}
