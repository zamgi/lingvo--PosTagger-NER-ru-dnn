using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>

    internal sealed class LSTMCell
    {
        private readonly WeightTensor _Wxh;
        private readonly WeightTensor _B;
        private WeightTensor _Hidden;
        private WeightTensor _Cell;
        private readonly int _Hdim;
        private readonly int _DeviceId;
        private readonly string _Name;
        private readonly LayerNormalization _LayerNorm1;
        private readonly LayerNormalization _LayerNorm2;

        public LSTMCell( string name, int hdim, int dim, int deviceId, bool isTrainable )
        {
            _Name = name;

            _Wxh = new WeightTensor( new long[ 2 ] { dim + hdim, hdim * 4 }, deviceId, normType: NormType.Uniform, name: $"{name}.m_Wxh", isTrainable: isTrainable );
            _B   = new WeightTensor( new long[ 2 ] { 1, hdim * 4 }, 0, deviceId, name: $"{name}.m_b", isTrainable: isTrainable );

            _Hdim     = hdim;
            _DeviceId = deviceId;

            _LayerNorm1 = new LayerNormalization( $"{name}.m_layerNorm1", hdim * 4, deviceId, isTrainable: isTrainable );
            _LayerNorm2 = new LayerNormalization( $"{name}.m_layerNorm2", hdim, deviceId, isTrainable: isTrainable );
        }

        public WeightTensor Hidden => _Hidden;

        public WeightTensor Step( WeightTensor input, ComputeGraphTensor g )
        {
            using ( ComputeGraphTensor innerGraph = g.CreateSubGraph( _Name ) )
            {
                WeightTensor hidden_prev = _Hidden;
                WeightTensor cell_prev   = _Cell;

                WeightTensor inputs = innerGraph.Concate( 1, input, hidden_prev );
                WeightTensor hhSum  = innerGraph.Affine( inputs, _Wxh, _B );
                WeightTensor hhSum2 = _LayerNorm1.Norm( innerGraph, hhSum );

                (WeightTensor gates_raw, WeightTensor cell_write_raw) = innerGraph.SplitColumns( hhSum2, _Hdim * 3, _Hdim );
                WeightTensor gates      = innerGraph.Sigmoid( gates_raw );
                WeightTensor cell_write = innerGraph.Tanh( cell_write_raw );

                (WeightTensor input_gate, WeightTensor forget_gate, WeightTensor output_gate) = innerGraph.SplitColumns( gates, _Hdim, _Hdim, _Hdim );

                // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
                _Cell = g.EltMulMulAdd( forget_gate, cell_prev, input_gate, cell_write );
                WeightTensor ct2 = _LayerNorm2.Norm( innerGraph, _Cell );

                // compute hidden state as gated, saturated cell activations
                _Hidden = g.EltMul( output_gate, innerGraph.Tanh( ct2 ) );

                return (_Hidden);
            }
        }
        public void Reset( WeightTensorFactory weightFactory, int batchSize )
        {
            if ( _Hidden != null )
            {
                _Hidden.Dispose();
                _Hidden = null;
            }
            if ( _Cell != null )
            {
                _Cell.Dispose();
                _Cell = null;
            }

            _Hidden = weightFactory.CreateWeightTensor( batchSize, _Hdim, _DeviceId, true, name: $"{_Name}.m_hidden", isTrainable: true );
            _Cell   = weightFactory.CreateWeightTensor( batchSize, _Hdim, _DeviceId, true, name: $"{_Name}.m_cell"  , isTrainable: true );
        }

        public List< WeightTensor > GetParams()
        {
            var response = new List< WeightTensor > { _Wxh, _B };

            response.AddRange( _LayerNorm1.GetParams() );
            response.AddRange( _LayerNorm2.GetParams() );

            return (response);
        }
        public void Save( Model model )
        {
            _Wxh.Save( model );
            _B.Save( model );

            _LayerNorm1.Save( model );
            _LayerNorm2.Save( model );
        }
        public void Load( Model model )
        {
            _Wxh.Load( model );
            _B.Load( model );

            _LayerNorm1.Load( model );
            _LayerNorm2.Load( model );
        }
    }
}
