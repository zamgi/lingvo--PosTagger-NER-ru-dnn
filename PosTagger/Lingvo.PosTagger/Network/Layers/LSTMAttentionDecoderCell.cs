using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>

    internal sealed class LSTMAttentionDecoderCell
    {
        private readonly int _HiddenDim;
        private readonly int _DeviceId;
        private readonly string _Name;
        private readonly WeightTensor _Wxhc;
        private readonly WeightTensor _B;
        private readonly LayerNormalization _LayerNorm1;
        private readonly LayerNormalization _LayerNorm2;

        public LSTMAttentionDecoderCell( string name, int hiddenDim, int inputDim, int contextDim, int deviceId, bool isTrainable )
        {
            _Name      = name;
            _HiddenDim = hiddenDim;
            _DeviceId  = deviceId;

            //---Logger.WriteLine( $"Create LSTM attention decoder cell '{name}' HiddemDim = '{hiddenDim}', InputDim = '{inputDim}', ContextDim = '{contextDim}', DeviceId = '{deviceId}'" );

            _Wxhc = new WeightTensor( new long[ 2 ] { inputDim + hiddenDim + contextDim, hiddenDim * 4 }, deviceId, normType: NormType.Uniform, name: $"{name}.m_Wxhc", isTrainable: isTrainable );
            _B    = new WeightTensor( new long[ 2 ] { 1, hiddenDim * 4 }, 0, deviceId, name: $"{name}.m_b", isTrainable: isTrainable );

            _LayerNorm1 = new LayerNormalization( $"{name}.m_layerNorm1", hiddenDim * 4, deviceId, isTrainable );
            _LayerNorm2 = new LayerNormalization( $"{name}.m_layerNorm2", hiddenDim, deviceId, isTrainable );
        }

        public WeightTensor Hidden { get; set; }
        public WeightTensor Cell   { get; set; }

        /// <summary>
        /// Update LSTM-Attention cells according to given weights
        /// </summary>
        /// <param name="context">The context weights for attention</param>
        /// <param name="input">The input weights</param>
        /// <param name="computeGraph">The compute graph to build workflow</param>
        /// <returns>Update hidden weights</returns>
        public WeightTensor Step( WeightTensor context, WeightTensor input, ComputeGraphTensor g )
        {
            using ( ComputeGraphTensor computeGraph = g.CreateSubGraph( _Name ) )
            {
                WeightTensor cell_prev   = Cell;
                WeightTensor hidden_prev = Hidden;

                WeightTensor hxhc   = computeGraph.Concate( 1, input, hidden_prev, context );
                WeightTensor hhSum  = computeGraph.Affine( hxhc, _Wxhc, _B );
                WeightTensor hhSum2 = _LayerNorm1.Norm( computeGraph, hhSum );

                (WeightTensor gates_raw, WeightTensor cell_write_raw) = computeGraph.SplitColumns( hhSum2, _HiddenDim * 3, _HiddenDim );
                WeightTensor gates      = computeGraph.Sigmoid( gates_raw );
                WeightTensor cell_write = computeGraph.Tanh( cell_write_raw );

                (WeightTensor input_gate, WeightTensor forget_gate, WeightTensor output_gate) = computeGraph.SplitColumns( gates, _HiddenDim, _HiddenDim, _HiddenDim );

                // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
                Cell = g.EltMulMulAdd( forget_gate, cell_prev, input_gate, cell_write );
                WeightTensor ct2 = _LayerNorm2.Norm( computeGraph, Cell );

                Hidden = g.EltMul( output_gate, computeGraph.Tanh( ct2 ) );

                return (Hidden);
            }
        }

        public List<WeightTensor> GetParams()
        {
            var response = new List<WeightTensor> { _Wxhc, _B };

            response.AddRange( _LayerNorm1.GetParams() );
            response.AddRange( _LayerNorm2.GetParams() );

            return (response);
        }

        public void Reset( WeightTensorFactory weightFactory, int batchSize )
        {
            Hidden = weightFactory.CreateWeightTensor( batchSize, _HiddenDim, _DeviceId, true, name: $"{_Name}.Hidden", isTrainable: true );
            Cell   = weightFactory.CreateWeightTensor( batchSize, _HiddenDim, _DeviceId, true, name: $"{_Name}.Cell", isTrainable: true );
        }

        public void Save( Model model )
        {
            _Wxhc.Save( model );
            _B.Save( model );

            _LayerNorm1.Save( model );
            _LayerNorm2.Save( model );
        }
        public void Load( Model model )
        {
            _Wxhc.Load( model );
            _B.Load( model );

            _LayerNorm1.Load( model );
            _LayerNorm2.Load( model );
        }
    }
}


