using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Optimizer;
using Lingvo.PosTagger.Text;
using Lingvo.PosTagger.Utils;

using Newtonsoft.Json;

namespace Lingvo.PosTagger.ModelBuilder
{
    /// <summary>
    /// 
    /// </summary>
    internal static class Program
    {
        private static void Main( string[] args )
        {
            try
            {
                var (opts, optsFileName) = OptionsExtensions.ReadInputOptions< Options >( args, 
                    _ => Logger.LogFile = $"Lingvo.PosTagger.ModelBuilder__({Misc.GetTimeStamp( DateTime.Now )}).log", 
                    "train.json" );

                //---test__OptionsFileChangingWatcher( opts, optsFileName/*@"D:\[git-svn]\[github]\lingvo\Lingvo.PosTagger_and_NER\ModelBuilding\Lingvo.PosTagger.ModelBuilder\train.json"*/ );
                //---test__creation_of_Corpus( opts );

                Run_Train( opts, optsFileName );
                //Validator.Run_Validate( opts );
            }
            catch ( Exception ex )
            {
                Logger.WriteErrorLine( Environment.NewLine + ex + Environment.NewLine );
            }
        }

        private static void test__ExternalValidatorRunner()
        {
            //Console.WriteLine( Environment.CurrentDirectory );
            //var r = (new ExternalValidatorRunner( opts.ExternalValidator )).ExternalValidateRoutine( default ); Console.WriteLine( r );
            //return;
        }
        private static void test__creation_of_Corpus( Options opts )
        {
            using var cts = Console_CancelKeyPress_Breaker.Create();

            using ( var trainCorpus = new Corpus( opts.TrainCorpusPath, opts.BatchSize, opts.MaxTrainSentLength, opts.TooLongSequence, cts.Token ) )
            {
                Console.WriteLine( trainCorpus.BatchSize );

                var sum = 0;
                foreach ( var batch in trainCorpus.GetBatchs( shuffle: true ) )
                {
                    sum += batch.SrcTotalTokensCount;
                }
                Console.WriteLine( sum );
            }
        }
        private static void test__OptionsFileChangingWatcher( Options initOpts, string optsFileName )
        {
            using var ofcw = new OptionsFileChangingWatcher( initOpts, optsFileName );

            var n = 1;
            ofcw.OnOptionsAllowedChanging += (in OptionsAllowedChanging opts) => Console.Title = $"[OptionsFileChangingWatcher: {++n}]";

            Console.ReadLine();
        }

        private static void Run_Train( Options opts, string optsFileName )
        {
            using var cts = Console_CancelKeyPress_Breaker.Create();

            // Load train corpus
            using var trainCorpus = new Corpus( opts.TrainCorpusPath, opts.BatchSize, opts.MaxTrainSentLength, opts.TooLongSequence, cts.Token );

            // Load valid corpus
            using var validCorpus = !opts.ValidCorpusPath.IsNullOrEmpty() ? new Corpus( opts.ValidCorpusPath, opts.BatchSize, opts.MaxPredictSentLength, opts.TooLongSequence, cts.Token ) : null;

            #region [.create SeqLabel instance.]
            SeqLabel sl;
            Vocab    tgtVocab;
            if ( File.Exists( opts.ModelFilePath ) )
            {
                //Incremental training
                Logger.WriteLine( $"Loading model from '{opts.ModelFilePath}'..." );
                sl = SeqLabel.Create4Train( opts );

                tgtVocab = sl.TgtVocab; //---trainCorpus.BuildTargetVocab( vocabIgnoreCase: false, opts.SrcVocabSize );
            }
            else
            {
                #region [.Load or build vocabulary.]
                Vocab srcVocab;
                if ( !opts.SrcVocab.IsNullOrEmpty() && !opts.TgtVocab.IsNullOrEmpty() )
                {
                    // Vocabulary files are specified, so we load them
                    srcVocab = new Vocab( opts.SrcVocab, ignoreCase: false );
                    tgtVocab = new Vocab( opts.TgtVocab, ignoreCase: false );
                }
                else
                {
                    // We don't specify vocabulary, so we build it from train corpus
                    (srcVocab, tgtVocab) = trainCorpus.BuildVocabs( vocabIgnoreCase: false, opts.SrcVocabSize );
                }
                #endregion

                //New training
                sl = SeqLabel.Create4Train( opts, srcVocab, tgtVocab );
            }
            #endregion

            #region [.ExternalValidatorRunner.]
            ExternalValidatorRunner evr = null;
            if ( !opts.ExternalValidator.FileName.IsNullOrWhiteSpace() )
            {
                Logger.WriteLine( $"Will be using External validator '{opts.ExternalValidator.FileName}'." );
                evr = new ExternalValidatorRunner( opts.ExternalValidator );
            }
            #endregion

            #region [.learningRate, optimizer, metrics.]
            // Create learning rate
            ILearningRate learningRate = new DecayLearningRate( opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount );

            // Create optimizer
            IOptimizer optimizer = Misc.CreateOptimizer( opts );

            // Create metrics
            var metrics = Validator.CreateFromTgtVocab_MultiLabelsFscoreMetric( tgtVocab );
            #endregion

            // Add event handler for monitoring
            sl.StatusUpdateWatcher += Misc.StatusUpdateWatcher;

            // Create options-file changing watcher
            using var fsw = OptionsFileChangingWatcher.TryCreate( opts, optsFileName, sl.OptionsWasChanging, trainCorpus.OptionsWasChanging, validCorpus.GetOptionsWasChangingRoutine() );

            // Start training
            sl.Train( trainCorpus, validCorpus, learningRate, metrics, optimizer, cancellationToken: cts.Token, evr.GetExternalValidateRoutine() );
        }
        private static ExternalValidateDelegate GetExternalValidateRoutine( this ExternalValidatorRunner evr ) => (evr != null) ? evr.ExternalValidateRoutine : null;
        private static OptionsAllowedChanging.ChangingDelege GetOptionsWasChangingRoutine( this Corpus corpus ) => (corpus != null) ? corpus.OptionsWasChanging : null;

        /// <summary>
        /// 
        /// </summary>
        private sealed class ExternalValidatorRunner
        {
            private string _FileName;
            private string _Arguments;
            private string _WorkingDirectory;
            public ExternalValidatorRunner( string fileName, string arguments, string workingDirectory )
            {
                _FileName  = fileName;
                _Arguments = arguments;
                _WorkingDirectory = workingDirectory ?? string.Empty;
            }
            public ExternalValidatorRunner( in Options.ExternalValidator_t opts )
            {
                _FileName = opts.FileName;
                _Arguments = opts.Arguments;
                _WorkingDirectory = opts.WorkingDirectory ?? string.Empty;
            }
            public Validator.Result ExternalValidateRoutine( CancellationToken ct )
            {
                try
                {
                    var pipeTask = PipeIPC.Server__in.RunDataReceiver( PipeIPC.PIPE_NAME_1, ct );

                    var runExternalValidateTask = Task.Run( () =>
                    {
                        var psi = new ProcessStartInfo( _FileName, _Arguments )
                        {
                            WorkingDirectory = _WorkingDirectory,
                            UseShellExecute  = true,
                            WindowStyle      = ProcessWindowStyle.Minimized //.Hidden
                        };
                        using ( var p = Process.Start( psi ) )
                        {
                            p.WaitForExit();
                        }
                    });
                    var waitCancelTask = Task.Delay( 1_000 * 60 * 30 /*30 min.*/ /*Timeout.Infinite*/, ct );

                    var tasks = new[] { pipeTask, runExternalValidateTask, waitCancelTask };
                    var taskIdx = Task.WaitAny( tasks, ct );
                    switch ( taskIdx )
                    {
                        case 0: //pipeTask
                            var (json, error) = pipeTask.Result;
                            if ( error != null )
                            {
                                throw (error);
                            }
                            var result = JsonConvert.DeserializeObject< Validator.Result >( json );
                            return (result);

                        case 1:
                            if ( runExternalValidateTask.Exception != null )
                            {
                                throw (runExternalValidateTask.Exception);
                            }
                            throw (new Exception( $"Error while running external validator. fn='{_FileName}', args='{_Arguments}'." ));

                        case 2:
                        default:
                            throw (new TimeoutException( $"Timeout while running external validator." ));
                    }
                }
                catch ( Exception ex )
                {
                    Logger.WriteErrorLine( "[EXTERNAL_VALIDATE_ROUTINE]: " + ex );
                    return (default);
                }            
            }
        }
    }
}
