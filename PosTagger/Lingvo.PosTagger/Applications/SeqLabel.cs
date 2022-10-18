using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Network;
using Lingvo.PosTagger.Text;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    public class SeqLabel : BaseSeq2SeqFramework< SeqLabelModel >
    {
        private MultiProcessorNetworkWrapper< WeightTensor >     _SrcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper< IEncoder >         _Encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper< FeedForwardLayer > _FFLayer; //The feed forward layers over over devices.
        private MultiProcessorNetworkWrapper< WeightTensor >    _PosEmbedding;
        private Options _Options;

        private SeqLabel( Options opts ) : base( opts )
        {
            if ( !File.Exists( opts.ModelFilePath ) ) throw (new FileNotFoundException( $"Model '{opts.ModelFilePath}' doesn't exist." ));

            _Options = opts;
            _Model   = LoadModel4Predict();

            _Model.ShowModelInfo();
        }
        private SeqLabel( Options opts, Vocab srcVocab, Vocab tgtVocab ) : base( opts )
        {
            _Options = opts;

            if ( File.Exists( _Options.ModelFilePath ) )
            {
                if ( (srcVocab != null) || (tgtVocab != null) )
                {
                    throw (new ArgumentException( $"Model '{_Options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null." ));
                }

                // Model file exists, so we load it from file.
                _Model = LoadModel4Train();
            }
            else
            {
                // Model doesn't exist, we create it and initlaize parameters
                _Model = new SeqLabelModel( opts.HiddenSize, opts.EmbeddingDim, opts.EncoderLayerDepth, opts.MultiHeadNum, opts.EncoderType, srcVocab, tgtVocab );

                //Initializng weights in encoders and decoders
                CreateTrainParameters( _Model );
            }

            _Model.ShowModelInfo();
        }
        public static SeqLabel Create4Predict( Options opts ) => new SeqLabel( opts );
        public static SeqLabel Create4Train( Options opts, Vocab srcVocab = null, Vocab tgtVocab = null ) => new SeqLabel( opts, srcVocab, tgtVocab );

        private SeqLabelModel LoadModel4Train() => base.LoadModelRoutine< Model_4_ProtoBufSerializer >( CreateTrainParameters, SeqLabelModel.Create );
        private SeqLabelModel LoadModel4Predict() => base.LoadModelRoutine< Model_4_ProtoBufSerializer >( CreatePredictParameters, SeqLabelModel.Create );
        private void CreateTrainParameters( Model model ) => CreateParametersRoutine( model, _Options.MaxTrainSentLength );
        private void CreatePredictParameters( Model model ) => CreateParametersRoutine( model, _Options.MaxPredictSentLength );
        private void CreateParametersRoutine( Model model, int maxSentLength )
        {
            Logger.WriteLine( $"Creating encoders and decoders..." );
            var deviceIds_Array = new RoundArray< int >( DeviceIds );

            var t = Encoder.CreateEncoders( model, _Options, deviceIds_Array );            
            var ffLayerWt = new FeedForwardLayer( "FeedForward", t.contextDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: deviceIds_Array.GetNextItem(), isTrainable: true );
            _FFLayer = new MultiProcessorNetworkWrapper< FeedForwardLayer >( ffLayerWt, DeviceIds );
            _Encoder = t.encoder;

            var srcEmbeddingWt = new WeightTensor( new long[ 2 ] { model.SrcVocab.Count, model.EncoderEmbeddingDim }, deviceIds_Array.GetNextItem(), normType: NormType.Uniform, name: "SrcEmbeddings", isTrainable: true );
            _SrcEmbedding = new MultiProcessorNetworkWrapper< WeightTensor >( srcEmbeddingWt, DeviceIds );

            if ( model.EncoderType == EncoderTypeEnums.Transformer )
            {
                var row    = maxSentLength + 2;
                var column = model.EncoderEmbeddingDim;
                var posEmbeddingWt = PositionEmbedding.BuildPositionWeightTensor( row, column, deviceIds_Array.GetNextItem(), "PosEmbedding", isTrainable: false );
                _PosEmbedding = new MultiProcessorNetworkWrapper< WeightTensor >( posEmbeddingWt, DeviceIds, isStaticWeights: true );
            }
            else
            {
                _PosEmbedding = null;
            }
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        private (IEncoder encoder, WeightTensor srcEmbedding, WeightTensor posEmbedding, FeedForwardLayer decoderFFLayer) GetNetworksOnDeviceAt( int deviceIdIdx ) 
            => (_Encoder.GetNetworkOnDevice( deviceIdIdx ), _SrcEmbedding.GetNetworkOnDevice( deviceIdIdx ), _PosEmbedding?.GetNetworkOnDevice( deviceIdIdx ), _FFLayer.GetNetworkOnDevice( deviceIdIdx ));

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="g">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side. In training mode, it inputs target tokens, otherwise, it outputs target tokens generated by decoder</param>
        /// <param name="deviceId">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        protected override NetworkResult RunForwardOnSingleDevice( ComputeGraphTensor g, CorpusBatch corpusBatch, int deviceId, bool isTraining, bool returnWordClassInfos = false )
        {
            var srcTokensBatch = corpusBatch.SrcTokensBatch;
            var tgtTokensBatch = corpusBatch.TgtTokensBatch;

            var (encoder, srcEmbedding, posEmbedding, decoderFFLayer) = GetNetworksOnDeviceAt( deviceId );

            var srcVocab = _Model.SrcVocab;
            var tgtVocab = _Model.TgtVocab;

            // Reset networks
            encoder.Reset( g.GetWeightFactory(), srcTokensBatch.Count );

            var originalSrcLengths = BuildInTokens.PadSentences( srcTokensBatch );
            var srcTokensList = srcVocab.GetIndicesByWords( srcTokensBatch );

            BuildInTokens.PadSentences_2( tgtTokensBatch ); //---BuildInTokens.PadSentences( tgtSnts );
            var seqLen    = srcTokensBatch[ 0 ].Count;
            var batchSize = srcTokensBatch.Count;

            // Encoding input source sentences
            WeightTensor encOutput = Encoder.Run( g/*, corpusBatch*/, encoder, _Model.EncoderType, srcEmbedding, posEmbedding/*, null*/, srcTokensList, originalSrcLengths );
            WeightTensor ffLayer   = decoderFFLayer.Process( g, encOutput, batchSize );

            float cost = 0.0f;
            var output_2 = default(List< NetworkResult.ClassesInfo >);            
            using ( WeightTensor probs = g.Softmax( ffLayer, inPlace: true, runGradients: false ) )
            {
                if ( isTraining )
                {
                    var indices = new long[ 2 ];

                    //Calculate loss for each word in the batch
                    for ( int k = 0; k < batchSize; k++ )
                    {
                        if ( tgtTokensBatch.Count <= k )
                        {
                            throw (new IndexOutOfRangeException( $"Sequence #'{k}' is out of range in target sequences (size '{tgtTokensBatch.Count})'. Source sequences batch size is '{srcTokensBatch.Count}'" ));
                        }

                        var tgtSnts__k = tgtTokensBatch[ k ];
                        for ( int j = 0; j < seqLen; j++ )
                        {
                            if ( tgtSnts__k.Count <= j )
                            {
                                var srcSnts__k = srcTokensBatch[ k ];
                                throw (new IndexOutOfRangeException( $"Token offset '{j}' is out of range in current target sequence (size = '{tgtSnts__k.Count}' text = '{string.Join( ' ', tgtSnts__k )}'). Source sequence size is '{srcSnts__k.Count}' text is {string.Join( ' ', srcSnts__k )}" ));
                            }

                            int ix_targets_k_j = tgtVocab.GetIndexByWord( tgtSnts__k[ j ] );
                            indices[ 0 ] = k * seqLen + j;
                            indices[ 1 ] = ix_targets_k_j;
                            float score_k = probs.GetWeightAt( indices/*new long[] { k * seqLen + j, ix_targets_k_j }*/ );
                            cost += (float) -Math.Log( score_k );

                            probs.SetWeightAt( score_k - 1, indices/*new long[] { k * seqLen + j, ix_targets_k_j }*/ );
                        }
                    }

                    ffLayer.CopyWeightsToGradients( probs );
                }
                else
                {
                    // Output "i"th target word
                    using var targetIdxTensor = g.Argmax( probs, 1 );
                    var targetIdx   = targetIdxTensor.ToWeightArray();
                    var targetWords = tgtVocab.GetWordsByIndices( targetIdx );

                    if ( (batchSize == 1) && returnWordClassInfos )
                    {
                        var probs_array  = probs.ToWeightArray();
                        output_2 = new List< NetworkResult.ClassesInfo >( batchSize );

                        Debug.Assert( probs.Columns == tgtVocab.Count );
                        Debug.Assert( probs.Rows    == targetWords.Count );

                        tgtTokensBatch[ 0 ] = targetWords;

                        var srcSnt = srcTokensBatch[ 0 ];
                        var cols = probs.Columns;
                        var wordCount = 0;
                        var inputBatchCount = srcSnt.Count;
                        var wordClasses = new List< NetworkResult.WordClassInfo >( inputBatchCount );
                        for ( int i = 0, len = inputBatchCount; i < len; i++ )
                        {
                            var word = srcSnt[ i ];

                            #region [.wordsInDictRatio.]
                            if ( BuildInTokens.IsPreDefinedToken( word ) )
                            {
                                inputBatchCount--;
                            }
                            else if ( srcVocab.ContainsWord( word ) )
                            {
                                wordCount++;
                            }
                            #endregion

                            var classes = new List< NetworkResult.ClassInfo >( cols - Vocab.START_MEANING_INDEX );                            
                            for ( int c = Vocab.START_MEANING_INDEX; c < cols; c++ )
                            {
                                var className = tgtVocab.GetWordByIndex( c );
                                var idx = i * cols + c;
                                var prob = probs_array[ idx ];
                                classes.Add( new NetworkResult.ClassInfo() { ClassName = className, Probability = prob } );
                            }
                            classes.Sort( NetworkResult.ClassInfo.ComparerByProbability.Inst );
                            wordClasses.Add( new NetworkResult.WordClassInfo() { Word = word, Classes = classes } );
                        }
                        var wordsInDictRatio = (0 < inputBatchCount) ? ((1.0f * wordCount) / inputBatchCount) : 0;
                        output_2.Add( new NetworkResult.ClassesInfo() { WordClasses = wordClasses, WordsInDictRatio = wordsInDictRatio } );
                    }
                    else
                    {
                        for ( int k = 0; k < batchSize; k++ )
                        {
                            tgtTokensBatch[ k ] = targetWords.GetRange( k * seqLen, seqLen );
                        }
                    }
                }
            }

            var nr = new NetworkResult( cost, tgtTokensBatch, output_2 );
            return (nr);
        }

        //--------------------------------------------------------------------//
        private static Predicate< string > _BuildInTokens_IsPreDefinedTokenPredicate = new Predicate< string >( s => string.IsNullOrEmpty( s ) || BuildInTokens.IsPreDefinedToken( s ) );
        private static Predicate< NetworkResult.WordClassInfo > _BuildInTokens_IsPreDefinedTokenPredicate_2 = new Predicate< NetworkResult.WordClassInfo >( t => string.IsNullOrEmpty( t.Word ) || BuildInTokens.IsPreDefinedToken( t.Word ) );
        public (List< string > labelTokens, NetworkResult.ClassesInfo classesInfos) Predict( List< string > inputTokens, bool returnWordClassInfos = false )
        {
            Debug.Assert( inputTokens.Count <= _Options.MaxPredictSentLength );

            var batch = new CorpusBatch( inputTokens );
            var nr    = RunPredictRoutine( batch, returnWordClassInfos );
            var labelTokens = nr.Output[ 0 ];
                labelTokens.RemoveAll( _BuildInTokens_IsPreDefinedTokenPredicate );
            var classesInfos = (nr.Output_2.AnyEx() ? nr.Output_2[ 0 ] : default);
                classesInfos.WordClasses?.RemoveAll( _BuildInTokens_IsPreDefinedTokenPredicate_2 );
            return (labelTokens, classesInfos);
        }
        private (List< string > labelTokens, NetworkResult.ClassesInfo classesInfos) Predict_Internal( List< string > inputTokens, bool returnWordClassInfos )
        {
            Debug.Assert( inputTokens.Count <= _Options.MaxPredictSentLength );

            var batch = new CorpusBatch( inputTokens );
            var nr    = RunPredictRoutine( batch, returnWordClassInfos );
            var labelTokens  = nr.Output[ 0 ];
            var classesInfos = (nr.Output_2.AnyEx() ? nr.Output_2[ 0 ] : default);
            return (labelTokens, classesInfos);
        }

        public (List< string > labelTokens, NetworkResult.ClassesInfo classesInfos) Predict_Full( List< string > inputTokens, int? maxPredictSentLength = null, float cutDropout = 0.1f, bool returnWordClassInfos = false )
        {
            var maxPredictSentLen = Math.Min( _Options.MaxPredictSentLength, maxPredictSentLength.GetValueOrDefault( _Options.MaxPredictSentLength ) );

            var d = inputTokens.Count - maxPredictSentLen;
            if ( 0 < d )
            {
                if ( (maxPredictSentLen * cutDropout) < d )
                {
                    return (Predict_Full_Routine( inputTokens, maxPredictSentLen, returnWordClassInfos ));
                }
                else
                {
                    inputTokens.RemoveRange( maxPredictSentLen/*inputTokens.Count - d*/, d );
                }
            }

            return (Predict( inputTokens, returnWordClassInfos ));
        }
        private (List< string > labelTokens, NetworkResult.ClassesInfo classesInfos) Predict_Full_Routine( List< string > inputTokens, int maxPredictSentLength, bool returnWordClassInfos )
        {
            var partCount    = Math.DivRem( inputTokens.Count, maxPredictSentLength, out var rem );
            var labelTokens  = new List< string >( inputTokens.Count );
            var wordClasses  = returnWordClassInfos ? new List< NetworkResult.WordClassInfo >() : default;
            var wordsInDictRatios = returnWordClassInfos ? new List< float >() : default;
            for ( var i = 0; i < partCount; i++ )
            {
                var words_part = inputTokens.GetRange( i * maxPredictSentLength, maxPredictSentLength );
                var (tt, ci) = Predict_Internal( words_part, returnWordClassInfos );
                labelTokens.AddRange( tt );
                if ( returnWordClassInfos )
                {
                    wordClasses.AddRange( ci.WordClasses );
                    wordsInDictRatios.Add( ci.WordsInDictRatio );
                }
            }
            if ( rem != 0 )
            {
                var words_part = inputTokens.GetRange( partCount * maxPredictSentLength, rem );
                var (tt, ci)   = Predict_Internal( words_part, returnWordClassInfos );
                labelTokens.AddRange( tt );
                if ( returnWordClassInfos )
                {
                    wordClasses.AddRange( ci.WordClasses );
                    wordsInDictRatios.Add( ci.WordsInDictRatio );
                }
            }

            labelTokens.RemoveAll( _BuildInTokens_IsPreDefinedTokenPredicate );
            wordClasses?.RemoveAll( _BuildInTokens_IsPreDefinedTokenPredicate_2 );
            var classesInfos = returnWordClassInfos ? new NetworkResult.ClassesInfo() { WordClasses = wordClasses, WordsInDictRatio = wordsInDictRatios.Sum() / wordsInDictRatios.Count } : default;
            return (labelTokens, classesInfos);
        }
        
        public void OptionsWasChanging( in OptionsAllowedChanging opts )
        {
            #region [.core of meaning.]
            if ( _Options.BatchSize             != opts.BatchSize             ) _Options.BatchSize             = opts.BatchSize;
            if ( _Options.MaxEpochNum           != opts.MaxEpochNum           ) _Options.MaxEpochNum           = opts.MaxEpochNum;
            if ( _Options.Valid_RunEveryUpdates != opts.Valid_RunEveryUpdates ) _Options.Valid_RunEveryUpdates = opts.Valid_RunEveryUpdates;
            #endregion
        }
    }
}
