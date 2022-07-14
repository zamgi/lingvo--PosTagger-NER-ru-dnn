using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class LSTMEncoder
    {
        private List<LSTMCell> _Encoders;        
        public LSTMEncoder( string name, int hdim, int dim, int depth, int deviceId, bool isTrainable )
        {
            _Encoders = new List<LSTMCell>( depth );
            _Encoders.Add( new LSTMCell( $"{name}.LSTM_0", hdim, dim, deviceId, isTrainable ) );
            for ( int i = 1; i < depth; i++ )
            {
                _Encoders.Add( new LSTMCell( $"{name}.LSTM_{i}", hdim, hdim, deviceId, isTrainable ) );
            }
        }

        public void Reset( WeightTensorFactory weightFactory, int batchSize )
        {
            foreach ( LSTMCell encoder in _Encoders )
            {
                encoder.Reset( weightFactory, batchSize );
            }
        }
        public WeightTensor Encode( WeightTensor V, ComputeGraphTensor g )
        {
            foreach ( LSTMCell encoder in _Encoders )
            {
                WeightTensor e = encoder.Step( V, g );
                V = e;
            }
            return (V);
        }

        public List<WeightTensor> GetParams()
        {
            var response = new List<WeightTensor>( _Encoders.Count );
            foreach ( var encoder in _Encoders )
            {
                response.AddRange( encoder.GetParams() );
            }
            return (response);
        }
        public void Save( Model model )
        {
            foreach ( var encoder in _Encoders )
            {
                encoder.Save( model );
            }
        }
        public void Load( Model model )
        {
            foreach ( var encoder in _Encoders )
            {
                encoder.Load( model );
            }
        }
    }
}
