using System;
using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class MultiHeadAttention
    {
        private readonly WeightTensor _W0;
        private readonly WeightTensor _B0;

        private readonly WeightTensor _Q;
        private readonly WeightTensor _K;
        private readonly WeightTensor _V;

        private readonly WeightTensor _Qb;
        private readonly WeightTensor _Kb;
        private readonly WeightTensor _Vb;

        private readonly WeightTensor _QKV;
        private readonly WeightTensor _QKVb;

        private readonly LayerNormalization _LayerNormQ;

        private readonly int    _HiddenDim;
        private readonly int    _D;
        private readonly int    _MultiHeadNum;
        private readonly string _Name;
        private readonly float  _DropoutRatio;
        private readonly bool   _SharedQKV;

        public MultiHeadAttention( string name, int multiHeadNum, int hiddenDim, int inputDim, float dropoutRatio, int deviceId, bool isTrainable, bool sharedQKV = false, float learningRateFactor = 1.0f )
        {
            _Name         = name;
            _HiddenDim    = hiddenDim;
            _MultiHeadNum = multiHeadNum;
            _D            = _HiddenDim / _MultiHeadNum;
            _DropoutRatio = dropoutRatio;
            _SharedQKV    = sharedQKV;

            _W0 = new WeightTensor( new long[ 2 ] { hiddenDim, hiddenDim }, deviceId, name: $"{name}.W0", isTrainable: isTrainable, normType: NormType.Uniform, learningRateFactor: learningRateFactor );
            _B0 = new WeightTensor( new long[ 2 ] { 1, hiddenDim }, 0, deviceId, name: $"{name}.b0", isTrainable: isTrainable );

            if ( !_SharedQKV )
            {
                _Q  = new WeightTensor( new long[ 2 ] { inputDim, hiddenDim }, deviceId, name: $"{name}.Q", isTrainable: isTrainable, normType: NormType.Uniform, learningRateFactor: learningRateFactor );
                _Qb = new WeightTensor( new long[ 2 ] { 1, hiddenDim }, 0, deviceId, name: $"{name}.Qb", isTrainable: isTrainable, learningRateFactor: learningRateFactor );

                _K  = new WeightTensor( new long[ 2 ] { inputDim, hiddenDim }, deviceId, name: $"{name}.K", isTrainable: isTrainable, normType: NormType.Uniform, learningRateFactor: learningRateFactor );
                _Kb = new WeightTensor( new long[ 2 ] { 1, hiddenDim }, 0, deviceId, name: $"{name}.Kb", isTrainable: isTrainable, learningRateFactor: learningRateFactor );

                _V  = new WeightTensor( new long[ 2 ] { inputDim, hiddenDim }, deviceId, name: $"{name}.V", isTrainable: isTrainable, normType: NormType.Uniform, learningRateFactor: learningRateFactor );
                _Vb = new WeightTensor( new long[ 2 ] { 1, hiddenDim }, 0, deviceId, name: $"{name}.Vb", isTrainable: isTrainable, learningRateFactor: learningRateFactor );
            }
            else
            {
                _QKV  = new WeightTensor( new long[ 2 ] { inputDim, hiddenDim * 3 }, deviceId, name: $"{name}.Q", isTrainable: isTrainable, normType: NormType.Uniform, learningRateFactor: learningRateFactor );
                _QKVb = new WeightTensor( new long[ 2 ] { 1, hiddenDim * 3 }, 0, deviceId, name: $"{name}.Qb", isTrainable: isTrainable, learningRateFactor: learningRateFactor );
            }

            _LayerNormQ = new LayerNormalization( $"{name}.layerNormQ", _HiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor );
        }

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="keyMask">The mask for softmax</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public (WeightTensor, WeightTensor) Perform( WeightTensor inputQ, WeightTensor keyMask, int batchSize, ComputeGraphTensor graph, bool outputAttenWeights = false )
        {
            using ComputeGraphTensor g = graph.CreateSubGraph( $"{_Name}_MultiHeadAttention" );
            int seqLenQ = inputQ.Rows / batchSize;

            WeightTensor inputQNorm = _LayerNormQ.Norm( inputQ, g );

            //Input projections
            var weightedQKV = g.View( g.Affine( inputQNorm, _QKV, _QKVb ), dims: new long[] { batchSize, seqLenQ, 3, _MultiHeadNum, _D } );
            var allQ = g.Select( weightedQKV, 2, 0 );
            var allK = g.Select( weightedQKV, 2, 1 );
            var allV = g.Select( weightedQKV, 2, 2 );


            //Multi-head attentions
            WeightTensor Qs = g.View( g.AsContiguous( g.Transpose( allQ, 1, 2 ) ), dims: new long[] { batchSize * _MultiHeadNum, seqLenQ, _D } );
            WeightTensor Ks = g.View( g.AsContiguous( g.Transpose( g.Transpose( allK, 1, 2 ), 2, 3 ) ), dims: new long[] { batchSize * _MultiHeadNum, _D, seqLenQ } );
            WeightTensor Vs = g.View( g.AsContiguous( g.Transpose( allV, 1, 2 ) ), dims: new long[] { batchSize * _MultiHeadNum, seqLenQ, _D } );

            // Scaled softmax
            float scale = 1.0f / (float) (Math.Sqrt( _D ));
            var   attn  = g.MulBatch( Qs, Ks, scale );
            attn = g.View( attn, dims: new long[] { batchSize, _MultiHeadNum, seqLenQ, seqLenQ } );

            if ( keyMask != null )
            {
                attn = g.Add( attn, keyMask, inPlace: true );
            }

            var attnProbs = g.Softmax( attn, inPlace: true );

            WeightTensor sumAttnWeights = null;
            if ( outputAttenWeights )
            {
                //Merge all attention probs over multi-heads
                sumAttnWeights = graph.Sum( attnProbs, 1 );
                sumAttnWeights = graph.Div( sumAttnWeights, (float) _MultiHeadNum );
                sumAttnWeights = graph.View( sumAttnWeights, new long[] { batchSize * seqLenQ, seqLenQ } );
            }

            attnProbs = g.View( attnProbs, dims: new long[] { batchSize * _MultiHeadNum, seqLenQ, seqLenQ } );

            WeightTensor o = g.View( g.MulBatch( attnProbs, Vs ), dims: new long[] { batchSize, _MultiHeadNum, seqLenQ, _D } );
            WeightTensor W = g.View( g.AsContiguous( g.Transpose( o, 1, 2 ) ), dims: new long[] { batchSize * seqLenQ, _MultiHeadNum * _D } );

            // Output projection
            WeightTensor finalAttResults = g.Dropout( g.Affine( W, _W0, _B0 ), batchSize, _DropoutRatio, inPlace: true );
            WeightTensor result = graph.Add( finalAttResults, inputQ, inPlace: true );

            return (result, sumAttnWeights);
        }

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="inputK">The input K tensor</param>
        /// <param name="inputV">The input V tensor</param>
        /// <param name="keyMask">The mask for softmax</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public (WeightTensor, WeightTensor) Perform( WeightTensor inputQ, WeightTensor inputK, WeightTensor inputV, WeightTensor keyMask, int batchSize, ComputeGraphTensor graph, bool outputAttenWeights = false, Dictionary<string, WeightTensor> cachedTensors = null )
        {
            var keyName = $"{_Name}_MultiHeadAttention";
            using ComputeGraphTensor g = graph.CreateSubGraph( keyName );
            int seqLenQ = inputQ.Rows / batchSize;

            // SeqLenK must be euqal to SeqLenV
            int seqLenK = inputK.Rows / batchSize;
            int seqLenV = inputV.Rows / batchSize;

            WeightTensor inputQNorm = _LayerNormQ.Norm( inputQ, g );

            //Input projections
            WeightTensor allQ = g.View( g.Affine( inputQNorm, _Q, _Qb ), dims: new long[] { batchSize, seqLenQ, _MultiHeadNum, _D } );
            WeightTensor allK = g.View( g.Affine( inputK,     _K, _Kb ), dims: new long[] { batchSize, seqLenK, _MultiHeadNum, _D } );
            WeightTensor allV = g.View( g.Affine( inputV,     _V, _Vb ), dims: new long[] { batchSize, seqLenV, _MultiHeadNum, _D } );

            //Multi-head attentions
            WeightTensor Qs = g.View( g.AsContiguous( g.Transpose( allQ, 1, 2 ) ), dims: new long[] { batchSize * _MultiHeadNum, seqLenQ, _D } );


            WeightTensor Ks = null;
            WeightTensor Vs = null;

            if ( cachedTensors == null ) // We don't use any cached tensors
            {
                Ks = g.View( g.AsContiguous( g.Transpose( g.Transpose( allK, 1, 2 ), 2, 3 ) ), dims: new long[] { batchSize * _MultiHeadNum, _D, seqLenK } );
                Vs = g.View( g.AsContiguous( g.Transpose( allV, 1, 2 ) ), dims: new long[] { batchSize * _MultiHeadNum, seqLenV, _D } );
            }
            else
            {
                var KsCacheName = keyName + "_Ks";
                var VsCacheName = keyName + "_Vs";

                if ( !cachedTensors.TryGetValue( KsCacheName, out Ks ) )
                {
                    Ks = g.View( g.AsContiguous( g.Transpose( g.Transpose( allK, 1, 2 ), 2, 3 ) ), dims: new long[] { batchSize * _MultiHeadNum, _D, seqLenK } );
                    cachedTensors.Add( KsCacheName, Ks.CopyWeightsRef( KsCacheName, Ks.NeedGradient ) );
                }
               
                if ( !cachedTensors.TryGetValue( VsCacheName, out Vs ) )
                {
                    Vs = g.View( g.AsContiguous( g.Transpose( allV, 1, 2 ) ), dims: new long[] { batchSize * _MultiHeadNum, seqLenV, _D } );
                    cachedTensors.Add( VsCacheName, Vs.CopyWeightsRef( VsCacheName, Vs.NeedGradient ) );
                }
            }

            // Scaled softmax
            float scale = 1.0f / (float) (Math.Sqrt( _D ));
            var attn = g.MulBatch( Qs, Ks, scale );
                attn = g.View( attn, dims: new long[] { batchSize, _MultiHeadNum, seqLenQ, seqLenK } );
            if ( keyMask != null )
            {
                attn = g.Add( attn, keyMask, inPlace: true );
            }

            var attnProbs = g.Softmax( attn, inPlace: true );

            WeightTensor sumAttnWeights = null;
            if ( outputAttenWeights )
            {
                sumAttnWeights = g.Select( attnProbs, 1, 0 );
                for ( int i = 1; i < _MultiHeadNum; i++ )
                {
                    var tmp = g.Select( attnProbs, 1, i );
                    sumAttnWeights = g.Add( sumAttnWeights, tmp );
                }

                sumAttnWeights = graph.Div( sumAttnWeights, (float) _MultiHeadNum );
                sumAttnWeights = graph.View( sumAttnWeights, new long[] { batchSize * seqLenQ, seqLenK } );
            }

            attnProbs = g.View( attnProbs, dims: new long[] { batchSize * _MultiHeadNum, seqLenQ, seqLenK } );

            WeightTensor o = g.View( g.MulBatch( attnProbs, Vs ), dims: new long[] { batchSize, _MultiHeadNum, seqLenQ, _D } );
            WeightTensor W = g.View( g.AsContiguous( g.Transpose( o, 1, 2 ) ), dims: new long[] { batchSize * seqLenQ, _MultiHeadNum * _D } );

            // Output projection
            WeightTensor finalAttResults = g.Dropout( g.Affine( W, _W0, _B0 ), batchSize, _DropoutRatio, inPlace: true );
            WeightTensor result = graph.Add( finalAttResults, inputQ, inPlace: true );

            return (result, sumAttnWeights);
        }

        public List< WeightTensor > GetParams()
        {
            var response = new List< WeightTensor > { _W0, _B0 };

            if ( !_SharedQKV )
            {
                response.Add( _Q );
                response.Add( _Qb );

                response.Add( _K );
                response.Add( _Kb );

                response.Add( _V );
                response.Add( _Vb );
            }
            else
            {
                response.Add( _QKV );
                response.Add( _QKVb );
            }

            response.AddRange( _LayerNormQ.GetParams() );

            return (response);
        }
        public void Save( Model model )
        {
            if ( !_SharedQKV )
            {
                _Q.Save( model );
                _Qb.Save( model );

                _K.Save( model );
                _Kb.Save( model );

                _V.Save( model );
                _Vb.Save( model );
            }
            else
            {
                _QKV.Save( model );
                _QKVb.Save( model );
            }

            _W0.Save( model );
            _B0.Save( model );

            _LayerNormQ.Save( model );
        }

        public void Load( Model model )
        {
            if ( !_SharedQKV )
            {
                _Q.Load( model );
                _Qb.Load( model );

                _K.Load( model );
                _Kb.Load( model );

                _V.Load( model );
                _Vb.Load( model );
            }
            else
            {
                _QKV.Load( model );
                _QKVb.Load( model );
            }

            _W0.Load( model );
            _B0.Load( model );

            _LayerNormQ.Load( model );
        }
    }
}
