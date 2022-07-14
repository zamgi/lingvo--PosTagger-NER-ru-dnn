using System;
using System.Collections.Generic;

using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class TransformerEncoder : IEncoder
    {
        private readonly List<MultiHeadAttention> _Encoders;
        private readonly List<PositionwiseFeedForward> _PosFFNs;

        private readonly int _InputDim;
        private readonly float _DropoutRatio;
        private readonly string _Name;
        private readonly int _MultiHeadNum;
        private readonly int _HiddenDim;
        private readonly int _Depth;
        private readonly int _DeviceId;
        private readonly bool _IsTrainable;
        private readonly float _LearningRateFactor;
        private readonly LayerNormalization _LayerNorm;

        public TransformerEncoder( string name, int multiHeadNum, int hiddenDim, int inputDim, int depth, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f )
        {
            Logger.WriteLine( $"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}'" );

            if ( hiddenDim != inputDim ) throw (new ArgumentException( $"hiddenDim is not equal to inputDim in TransformerEncoder." ));

            _Name               = name;
            _MultiHeadNum       = multiHeadNum;
            _HiddenDim          = hiddenDim;
            _InputDim           = inputDim;
            _Depth              = depth;
            _DropoutRatio       = dropoutRatio;
            _DeviceId           = deviceId;
            _IsTrainable        = isTrainable;
            _LearningRateFactor = learningRateFactor;

            _Encoders = new List<MultiHeadAttention>( depth );
            _Encoders.Add( new MultiHeadAttention( $"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, _DropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor ) );
            for ( int i = 1; i < depth; i++ )
            {
                _Encoders.Add( new MultiHeadAttention( $"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, _DropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor ) );
            }

            _PosFFNs = new List<PositionwiseFeedForward>( depth );
            for ( int i = 0; i < depth; i++ )
            {
                _PosFFNs.Add( new PositionwiseFeedForward( $"{name}.PosFFN_{i}", hiddenDim, _DropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor ) );
            }

            _LayerNorm = new LayerNormalization( $"{name}.layerNorm", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor );
        }

        public int GetDeviceId() => _DeviceId;
        public void Reset( WeightTensorFactory weightFactory, int batchSize ) { }

        public WeightTensor Encode( WeightTensor inputs, int batchSize, ComputeGraphTensor g, WeightTensor srcSelfMask )
        {
            using ( ComputeGraphTensor subg = g.CreateSubGraph( $"{_Name}_Encoder" ) )
            {
                WeightTensor maskTensor = null;
                if ( srcSelfMask != null )
                {
                    int seqLen = inputs.Rows / batchSize;
                    using var keyMaskView = subg.View( srcSelfMask, dims: new long[] { batchSize, 1, seqLen, seqLen } );
                    maskTensor = subg.Expand( keyMaskView, dims: new long[] { batchSize, _MultiHeadNum, seqLen, seqLen } );
                }

                WeightTensor attnProbs = null;
                for ( int k = 0; k < _Encoders.Count; k++ )
                {
                    (inputs, attnProbs) = _Encoders[ k ].Perform( inputs, maskTensor, batchSize, subg, outputAttenWeights: false );
                    inputs = _PosFFNs[ k ].Perform( inputs, batchSize, subg );
                }
                inputs = _LayerNorm.Norm( inputs, subg );

                inputs.UnbindFromComputeGraph();
                attnProbs?.UnbindFromComputeGraph();
                maskTensor?.Dispose();
            }
            return (inputs);
        }

        public INeuralUnit CloneToDeviceAt( int deviceId ) => new TransformerEncoder( _Name, _MultiHeadNum, _HiddenDim, _InputDim, _Depth, _DropoutRatio, deviceId, _IsTrainable, learningRateFactor: _LearningRateFactor );
        public List<WeightTensor> GetParams()
        {
            var response = new List< WeightTensor >( _Encoders.Count * 10 + _PosFFNs.Count * 10 + 1 );
            foreach ( var e in _Encoders )
            {
                response.AddRange( e.GetParams() );
            }
            foreach ( var p in _PosFFNs )
            {
                response.AddRange( p.GetParams() );
            }

            response.AddRange( _LayerNorm.GetParams() );
            return (response);
        }

        public void Save( Model model )
        {
            foreach ( var e in _Encoders )
            {
                e.Save( model );
            }
            foreach ( var p in _PosFFNs )
            {
                p.Save( model );
            }
            _LayerNorm.Save( model );
        }

        public void Load( Model model )
        {
            foreach ( var e in _Encoders )
            {
                e.Load( model );
            }
            foreach ( var p in _PosFFNs )
            {
                p.Load( model );
            }
            _LayerNorm.Load( model );
        }
    }
}
