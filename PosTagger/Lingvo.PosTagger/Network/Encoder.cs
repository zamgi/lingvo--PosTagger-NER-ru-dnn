using System;
using System.Collections.Generic;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal static class Encoder
    {
        public static (MultiProcessorNetworkWrapper< IEncoder > encoder, int contextDim) CreateEncoders( Model model, Options opts, RoundArray< int > deviceIds )
        {
            int contextDim;
            MultiProcessorNetworkWrapper< IEncoder > encoder;
            if ( model.EncoderType == EncoderTypeEnums.BiLSTM )
            {
                encoder = new MultiProcessorNetworkWrapper< IEncoder >(
                    new BiEncoder( "BiLSTMEncoder", model.HiddenDim, model.EncoderEmbeddingDim, model.EncoderLayerDepth, deviceIds.GetNextItem(), isTrainable: opts.IsEncoderTrainable ), deviceIds.ToArray() );

                contextDim = model.HiddenDim * 2;
            }
            else
            {
                encoder = new MultiProcessorNetworkWrapper< IEncoder >(
                    new TransformerEncoder( "TransformerEncoder", model.MultiHeadNum, model.HiddenDim, model.EncoderEmbeddingDim, model.EncoderLayerDepth, opts.DropoutRatio, deviceIds.GetNextItem(),
                    isTrainable: opts.IsEncoderTrainable, learningRateFactor: opts.EncoderStartLearningRateFactor ), deviceIds.ToArray() );

                contextDim = model.HiddenDim;
            }
            return (encoder, contextDim);
        }

        public static WeightTensor Run( ComputeGraphTensor computeGraph, /*CorpusBatch corpusBatch,*/ IEncoder encoder, EncoderTypeEnums modelEncoderType,
            WeightTensor srcEmbedding, WeightTensor posEmbedding/*, WeightTensor segmentEmbedding*/, List< List< int > > srcSntsIds, float[] originalSrcLengths )
        {
            // Reset networks
            encoder.Reset( computeGraph.GetWeightFactory(), srcSntsIds.Count );

            #region comm. Build contextual feature if they exist. (never more than 1 group).
            /*
            WeightTensor contextTensor = null;
            for ( int i = 1, cnt = corpusBatch.GetSrcGroupSize(); i < cnt; i++ )
            {
                var contextCLSOutput = BuildTensorForSourceTokenGroupAt( computeGraph, corpusBatch, encoder, model, srcEmbedding, posEmbedding, segmentEmbedding, i );
                if ( contextTensor == null )
                {
                    contextTensor = contextCLSOutput;
                }
                else
                {
                    contextTensor = computeGraph.Add( contextTensor, contextCLSOutput );
                }
            }
            //*/
            #endregion

            WeightTensor encOutput = InnerRunner( computeGraph, srcSntsIds, originalSrcLengths, encoder, modelEncoderType, srcEmbedding, posEmbedding/*, segmentEmbedding*/ /*, contextTensor*/ );
            return (encOutput);
        }

        //public static WeightTensor BuildTensorForSourceTokenGroupAt( ComputeGraphTensor computeGraph, CorpusBatch corpusBatch, IEncoder encoder, Model model, WeightTensor srcEmbedding, WeightTensor posEmbedding, WeightTensor segmentEmbedding, int groupId )
        //{
        //    var contextTokens            = InsertCLSToken( corpusBatch.GetSrcTokens( groupId ) );
        //    var originalSrcContextLength = BuildInTokens.PadSentences( contextTokens );
        //    var contextTokenIds          = model.SrcVocab.GetWordIndex( contextTokens );

        //    WeightTensor encContextOutput = InnerRunner( computeGraph, contextTokenIds, originalSrcContextLength, encoder, model, srcEmbedding, posEmbedding, segmentEmbedding );

        //    var contextPaddedLen = contextTokens[ 0 ].Count;
        //    var contextCLSIdxs   = new float[ corpusBatch.GetBatchSize() ];
        //    for ( int j = 0; j < contextCLSIdxs.Length; j++ )
        //    {
        //        contextCLSIdxs[ j ] = j * contextPaddedLen;
        //    }

        //    WeightTensor contextCLSOutput = computeGraph.IndexSelect( encContextOutput, contextCLSIdxs );
        //    return (contextCLSOutput);
        //}

        private static WeightTensor InnerRunner( ComputeGraphTensor computeGraph, List< List< int > > srcTokensList, float[] originalSrcLengths, IEncoder encoder, EncoderTypeEnums modelEncoderType,
           WeightTensor srcEmbedding, WeightTensor posEmbedding/*, WeightTensor segmentEmbedding*/ /*, WeightTensor contextEmbeddings = null*/ )
        {
            var batchSize       = srcTokensList.Count;
            var srcSeqPaddedLen = srcTokensList[ 0 ].Count;

            // The length of source sentences are same in a single mini-batch, so we don't have source mask.
            using ( var srcSelfMask = (1 < batchSize) ? computeGraph.BuildPadSelfMask( srcSeqPaddedLen, originalSrcLengths ) : null )
            {
                // Encoding input source sentences
                var encOutput = RunEncoder( computeGraph, srcTokensList, encoder, modelEncoderType, srcEmbedding, srcSelfMask, posEmbedding/*, segmentEmbedding*/ /*, contextEmbeddings*/ );
                return (encOutput);
            }
        }

        /// <summary>
        /// Encode source sentences and output encoded weights
        /// </summary>
        private static WeightTensor RunEncoder( ComputeGraphTensor g, List< List< int > > seqs, IEncoder encoder, EncoderTypeEnums modelEncoderType, WeightTensor embeddings, WeightTensor selfMask, WeightTensor posEmbeddings
            /*, WeightTensor segmentEmbeddings*//*, WeightTensor contextEmbeddings*/ )
        {
            var batchSize = seqs.Count;
            var inputEmbs = TensorUtils.CreateTokensEmbeddings( seqs, g, embeddings/*, segmentEmbeddings*/ /*, contextEmbeddings*//*, model.SrcVocab*//*, model.ApplyContextEmbeddingsToEntireSequence*/, (float) Math.Sqrt( embeddings.Columns ) );

            if ( modelEncoderType == EncoderTypeEnums.Transformer )
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding( g, posEmbeddings, batchSize, inputEmbs, 0.0f );
            }

            var encOutput = encoder.Encode( inputEmbs, batchSize, g, selfMask );
            return (encOutput);
        }
    }
}
