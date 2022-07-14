using System.Collections.Generic;
using System.Linq;

using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class BiEncoder : IEncoder
    {
        private readonly List<LSTMCell> _ForwardEncoders;
        private readonly List<LSTMCell> _BackwardEncoders;
        private readonly string _Name;
        private readonly int _HiddenDim;
        private readonly int _InputDim;
        private readonly int _Depth;
        private readonly int _DeviceId;
        private readonly bool _IsTrainable;

        public BiEncoder( string name, int hiddenDim, int inputDim, int depth, int deviceId, bool isTrainable )
        {
            Logger.WriteLine( $"Creating BiLSTM encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', IsTrainable = '{isTrainable}'" );

            _ForwardEncoders  = new List<LSTMCell>( depth );
            _BackwardEncoders = new List<LSTMCell>( depth );

            _ForwardEncoders.Add( new LSTMCell( $"{name}.Forward_LSTM_0", hiddenDim, inputDim, deviceId, isTrainable: isTrainable ) );
            _BackwardEncoders.Add( new LSTMCell( $"{name}.Backward_LSTM_0", hiddenDim, inputDim, deviceId, isTrainable: isTrainable ) );

            for ( int i = 1; i < depth; i++ )
            {
                _ForwardEncoders.Add( new LSTMCell( $"{name}.Forward_LSTM_{i}", hiddenDim, hiddenDim * 2, deviceId, isTrainable: isTrainable ) );
                _BackwardEncoders.Add( new LSTMCell( $"{name}.Backward_LSTM_{i}", hiddenDim, hiddenDim * 2, deviceId, isTrainable: isTrainable ) );
            }

            _Name        = name;
            _HiddenDim   = hiddenDim;
            _InputDim    = inputDim;
            _Depth       = depth;
            _DeviceId    = deviceId;
            _IsTrainable = isTrainable;
        }

        public int GetDeviceId() => _DeviceId;
        public INeuralUnit CloneToDeviceAt( int deviceId ) => new BiEncoder( _Name, _HiddenDim, _InputDim, _Depth, deviceId, _IsTrainable );

        public void Reset( WeightTensorFactory weightFactory, int batchSize )
        {
            foreach ( var c in _ForwardEncoders )
            {
                c.Reset( weightFactory, batchSize );
            }
            foreach ( var c in _BackwardEncoders )
            {
                c.Reset( weightFactory, batchSize );
            }
        }

        public WeightTensor Encode( WeightTensor rawInputs, int batchSize, ComputeGraphTensor g, WeightTensor srcSelfMask )
        {
            int seqLen = rawInputs.Rows / batchSize;

            rawInputs = g.TransposeBatch( rawInputs, seqLen );

            var inputs = new List<WeightTensor>( seqLen );
            for ( int i = 0; i < seqLen; i++ )
            {
                WeightTensor emb_i = g.Peek( rawInputs, 0, i * batchSize, batchSize );
                inputs.Add( emb_i );
            }

            var forwardOutputs  = new List<WeightTensor>( _Depth * seqLen );
            var backwardOutputs = new List<WeightTensor>( _Depth * seqLen );

            List<WeightTensor> layerOutputs = inputs.ToList();
            for ( int i = 0; i < _Depth; i++ )
            {
                for ( int j = 0; j < seqLen; j++ )
                {
                    WeightTensor forwardOutput = _ForwardEncoders[ i ].Step( layerOutputs[ j ], g );
                    forwardOutputs.Add( forwardOutput );

                    WeightTensor backwardOutput = _BackwardEncoders[ i ].Step( layerOutputs[ inputs.Count - j - 1 ], g );
                    backwardOutputs.Add( backwardOutput );
                }

                backwardOutputs.Reverse();
                layerOutputs.Clear();
                for ( int j = 0; j < seqLen; j++ )
                {
                    WeightTensor concatW = g.Concate( 1, forwardOutputs[ j ], backwardOutputs[ j ] );
                    layerOutputs.Add( concatW );
                }

            }

            var result = g.Concate( layerOutputs, 0 );

            return (g.TransposeBatch( result, batchSize ));
        }

        public List<WeightTensor> GetParams()
        {
            var response = new List<WeightTensor>( _ForwardEncoders.Count * 10 + _BackwardEncoders.Count * 10 );
            foreach ( var c in _ForwardEncoders )
            {
                response.AddRange( c.GetParams() );
            }
            foreach ( var c in _BackwardEncoders )
            {
                response.AddRange( c.GetParams() );
            }
            return (response);
        }
        public void Save( Model model )
        {
            foreach ( var c in _ForwardEncoders )
            {
                c.Save( model );
            }
            foreach ( var c in _BackwardEncoders )
            {
                c.Save( model );
            }
        }
        public void Load( Model model )
        {
            foreach ( var c in _ForwardEncoders )
            {
                c.Load( model );
            }
            foreach ( var c in _BackwardEncoders )
            {
                c.Load( model );
            }
        }
    }
}
