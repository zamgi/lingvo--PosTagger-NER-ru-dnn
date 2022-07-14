using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class AttentionPreProcessResult
    {
        public WeightTensor Uhs;
        public WeightTensor encOutput;
    }
    /// <summary>
    /// 
    /// </summary>

    internal sealed class AttentionUnit : INeuralUnit
    {
        private readonly WeightTensor _V;
        private readonly WeightTensor _Ua;
        private readonly WeightTensor _bUa;
        private readonly WeightTensor _Wa;
        private readonly WeightTensor _bWa;

        private readonly string _Nname;
        private readonly int _HiddenDim;
        private readonly int _ContextDim;
        private readonly int _DeviceId;
        private readonly bool _IsTrainable;

        private bool _EnableCoverageModel;
        private readonly WeightTensor _Wc;
        private readonly WeightTensor _bWc;
        private readonly LSTMCell _Coverage;

        private readonly int _CoverageModelDim = 16;

        public AttentionUnit( string name, int hiddenDim, int contextDim, int deviceId, bool enableCoverageModel, bool isTrainable )
        {
            _Nname                = name;
            _HiddenDim           = hiddenDim;
            _ContextDim          = contextDim;
            _DeviceId            = deviceId;
            _EnableCoverageModel = enableCoverageModel;
            _IsTrainable         = isTrainable;

            //---Logger.WriteLine( $"Creating attention unit '{name}' HiddenDim = '{hiddenDim}', ContextDim = '{contextDim}', DeviceId = '{deviceId}', EnableCoverageModel = '{enableCoverageModel}'" );

            _Ua  = new WeightTensor( new long[ 2 ] { contextDim, hiddenDim }, deviceId, normType: NormType.Uniform, name: $"{name}.m_Ua", isTrainable: isTrainable );
            _Wa  = new WeightTensor( new long[ 2 ] { hiddenDim, hiddenDim }, deviceId, normType: NormType.Uniform, name: $"{name}.m_Wa", isTrainable: isTrainable );
            _bUa = new WeightTensor( new long[ 2 ] { 1, hiddenDim }, 0, deviceId, name: $"{name}.m_bUa", isTrainable: isTrainable );
            _bWa = new WeightTensor( new long[ 2 ] { 1, hiddenDim }, 0, deviceId, name: $"{name}.m_bWa", isTrainable: isTrainable );
            _V   = new WeightTensor( new long[ 2 ] { hiddenDim, 1 }, deviceId, normType: NormType.Uniform, name: $"{name}.m_V", isTrainable: isTrainable );

            if ( _EnableCoverageModel )
            {
                _Wc       = new WeightTensor( new long[ 2 ] { _CoverageModelDim, hiddenDim }, deviceId, normType: NormType.Uniform, name: $"{name}.m_Wc", isTrainable: isTrainable );
                _bWc      = new WeightTensor( new long[ 2 ] { 1, hiddenDim }, 0, deviceId, name: $"{name}.m_bWc", isTrainable: isTrainable );
                _Coverage = new LSTMCell( name: $"{name}.m_coverage", hdim: _CoverageModelDim, dim: 1 + contextDim + hiddenDim, deviceId: deviceId, isTrainable: isTrainable );
            }
        }

        public int GetDeviceId() => _DeviceId;

        public AttentionPreProcessResult PreProcess( WeightTensor encOutput, int batchSize, ComputeGraphTensor g )
        {
            int srcSeqLen = encOutput.Rows / batchSize;

            var r = new AttentionPreProcessResult() { encOutput = encOutput };
            r.Uhs = g.Affine( r.encOutput, _Ua, _bUa );
            r.Uhs = g.View( r.Uhs, dims: new long[] { batchSize, srcSeqLen, -1 } );
            if ( _EnableCoverageModel )
            {
                _Coverage.Reset( g.GetWeightFactory(), r.encOutput.Rows );
            }
            return (r);
        }

        public WeightTensor Perform( WeightTensor state, AttentionPreProcessResult attnPre, int batchSize, ComputeGraphTensor graph )
        {
            int srcSeqLen = attnPre.encOutput.Rows / batchSize;

            using ( ComputeGraphTensor g = graph.CreateSubGraph( _Nname ) )
            {
                // Affine decoder state
                WeightTensor wc = g.Affine( state, _Wa, _bWa );

                // Expand dims from [batchSize x decoder_dim] to [batchSize x srcSeqLen x decoder_dim]
                WeightTensor wc1   = g.View( wc, dims: new long[] { batchSize, 1, wc.Columns } );
                WeightTensor wcExp = g.Expand( wc1, dims: new long[] { batchSize, srcSeqLen, wc.Columns } );

                WeightTensor ggs = null;
                if ( _EnableCoverageModel )
                {
                    // Get coverage model status at {t-1}
                    WeightTensor wCoverage  = g.Affine( _Coverage.Hidden, _Wc, _bWc );
                    WeightTensor wCoverage1 = g.View( wCoverage, dims: new long[] { batchSize, srcSeqLen, -1 } );

                    ggs = g.AddTanh( attnPre.Uhs, wcExp, wCoverage1 );
                }
                else
                {
                    ggs = g.AddTanh( attnPre.Uhs, wcExp );
                }

                WeightTensor ggss  = g.View( ggs, dims: new long[] { batchSize * srcSeqLen, -1 } );
                WeightTensor atten = g.Mul( ggss, _V );

                WeightTensor attenT  = g.Transpose( atten );
                WeightTensor attenT2 = g.View( attenT, dims: new long[] { batchSize, srcSeqLen } );

                WeightTensor attenSoftmax1 = g.Softmax( attenT2, inPlace: true );

                WeightTensor attenSoftmax = g.View( attenSoftmax1, dims: new long[] { batchSize, 1, srcSeqLen } );
                WeightTensor inputs2      = g.View( attnPre.encOutput, dims: new long[] { batchSize, srcSeqLen, attnPre.encOutput.Columns } );

                WeightTensor contexts = graph.MulBatch( attenSoftmax, inputs2 );

                contexts = graph.View( contexts, dims: new long[] { batchSize, attnPre.encOutput.Columns } );

                if ( _EnableCoverageModel )
                {
                    // Concatenate tensor as input for coverage model
                    WeightTensor aCoverage = g.View( attenSoftmax1, dims: new long[] { attnPre.encOutput.Rows, 1 } );

                    WeightTensor state2 = g.View  ( state, dims: new long[] { batchSize, 1, state.Columns } );
                    WeightTensor state3 = g.Expand( state2, dims: new long[] { batchSize, srcSeqLen, state.Columns } );
                    WeightTensor state4 = g.View  ( state3, dims: new long[] { batchSize * srcSeqLen, -1 } );

                    WeightTensor concate = g.Concate( 1, aCoverage, attnPre.encOutput, state4 );
                    _Coverage.Step( concate, graph );
                }

                return (contexts);
            }
        }

        public List<WeightTensor> GetParams()
        {
            var response = new List<WeightTensor>
            {
                _Ua,
                _Wa,
                _bUa,
                _bWa,
                _V
            };

            if ( _EnableCoverageModel )
            {
                response.Add( _Wc );
                response.Add( _bWc );
                response.AddRange( _Coverage.GetParams() );
            }

            return (response);
        }

        public void Save( Model model )
        {
            _Ua.Save( model );
            _Wa.Save( model );
            _bUa.Save( model );
            _bWa.Save( model );
            _V.Save( model );

            if ( _EnableCoverageModel )
            {
                _Wc.Save( model );
                _bWc.Save( model );
                _Coverage.Save( model );
            }
        }
        public void Load( Model model )
        {
            _Ua.Load( model );
            _Wa.Load( model );
            _bUa.Load( model );
            _bWa.Load( model );
            _V.Load( model );

            if ( _EnableCoverageModel )
            {
                _Wc.Load( model );
                _bWc.Load( model );
                _Coverage.Load( model );
            }
        }

        public INeuralUnit CloneToDeviceAt( int deviceId ) => new AttentionUnit( _Nname, _HiddenDim, _ContextDim, deviceId, _EnableCoverageModel, _IsTrainable );
    }
}



