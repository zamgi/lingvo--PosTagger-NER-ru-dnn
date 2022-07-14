using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using Newtonsoft.Json;

using Lingvo.PosTagger.Tokenizing;
using Lingvo.PosTagger.Utils;
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.WebService
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class ConcurrentFactory : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly struct ModelInfoConfig
        {
            public Predictor  predictor        { get; init; }
            public string     modelFilePath    { get; init; }
            public Tokenizer  tokenizer        { get; init; }
            public int        maxEndingLength  { get; init; }
            public string     regimenModelType { get; init; }
        }

        /// <summary>
        /// 
        /// </summary>
        private sealed class ModelTuple
        {
            public ModelTuple( in ModelInfoConfig t )
            {
                WeakRef          = new WeakReference< Predictor >( t.predictor );
                ModelFilePath    = t.modelFilePath;
                Tokenizer        = t.tokenizer;
                MaxEndingLength  = t.maxEndingLength;
                RegimenModelType = t.regimenModelType;
            }
            public WeakReference< Predictor > WeakRef          { get; }
            public string                     ModelFilePath    { get; }
            public Tokenizer                  Tokenizer        { get; }
            public int                        MaxEndingLength  { get; }
            public string                     RegimenModelType { get; }
        }

        #region [.ctor().]
        private SemaphoreSlim _Semaphore;
        private Config        _Opts;
        private IReadOnlyDictionary< string, ModelTuple > _SLByType;
        private bool                 _EnableLog;
        private string               _LogFileName;
        private AsyncCriticalSection _LogCS;
        public ConcurrentFactory( IReadOnlyDictionary< string, ModelInfoConfig > slByType, Config opts, int instanceCount )
        {
            if ( instanceCount <= 0 ) throw (new ArgumentException( nameof(instanceCount) ));
            if ( slByType == null )   throw (new ArgumentException( nameof(slByType) ));
            if ( !slByType.Any() )    throw (new ArgumentException( nameof(slByType) ));
            if ( opts == null )       throw (new ArgumentException( nameof(opts) ));
            //------------------------------------------------------------------------------------------------------//

            _Opts      = opts;
            _Semaphore = new SemaphoreSlim( instanceCount, instanceCount );
            _SLByType  = slByType.ToDictionary( p => p.Key, p => new ModelTuple( p.Value ) );
            _EnableLog = _Opts.Log.Enable;
            if ( _EnableLog )
            {
                _LogFileName = Path.GetFullPath( _Opts.Log.LogFileName );
                _LogCS       = AsyncCriticalSection.Create();
            }
        }
        public void Dispose()
        {
            _Semaphore.Dispose();
            _LogCS    .Dispose();
        }
        #endregion

        private static List< (IList< ResultVM.TupleVM > result, Exception error) > EMPTY = new List< (IList< ResultVM.TupleVM > result, Exception error) >();
        private static Exception EMPTY_Exception = new Exception( "EMPTY text" );
        [M(O.AggressiveInlining)] private Predictor GetPredictor( ModelTuple t )
        {
            if ( !t.WeakRef.TryGetTarget( out var predictor ) || (predictor == null) )
            {
                lock ( t.WeakRef )
                {
                    if ( !t.WeakRef.TryGetTarget( out predictor ) || (predictor == null) )
                    {
                        var opts = JsonConvert.DeserializeObject< Config >( JsonConvert.SerializeObject( _Opts ) );
                        opts.ModelFilePath = t.ModelFilePath;

                        var sl = SeqLabel.Create4Predict( opts );
                        predictor = new Predictor( sl );
                        t.WeakRef.SetTarget( predictor );
                    }
                }
            }
            return (predictor);
        }
        [M(O.AggressiveInlining)] private ModelTuple GetModelTuple( string modelType, bool getFirstModelTypeIfMissing )
        {
            ModelTuple mt;
            if ( getFirstModelTypeIfMissing )
            {
                if ( modelType.IsNullOrEmpty() ) mt = _SLByType.Values.First();
                else if ( !_SLByType.TryGetValue( modelType, out mt ) ) throw (new ArgumentNullException( nameof(modelType) ));
            }
            else
            {
                if ( modelType.IsNullOrEmpty() ) throw (new ArgumentNullException( nameof(modelType) ));
                if ( !_SLByType.TryGetValue( modelType, out mt ) ) throw (new ArgumentNullException( nameof(modelType) ));
            }
            return (mt);
        }
        [M(O.AggressiveInlining)] private static IList< ResultVM.TupleVM > CreateResultTuples( List< word_t > input_words, IList< string > output_words, string text
            , bool setPosTaggerOutputType = true )
        {
            if ( setPosTaggerOutputType )
            {
                input_words.SetPosTaggerOutputType( output_words );

                var len = input_words.Count;
                var res = new ResultVM.TupleVM[ len ];
                for ( var i = 0; i < len; i++ )
                {
                    var w = input_words[ i ];
                    res[ i ]= new ResultVM.TupleVM() { Word = text.Substring( w.startIndex, w.length ), Label = w.seqLabelOutputType/*posTaggerOutputType.ToText()*/ };
                }
                return (res);
            }
            else
            {
#if DEBUG
                Debug.Assert( input_words.Count == output_words.Count );
#endif
                var len = Math.Min( input_words.Count, output_words.Count );
                var res = new ResultVM.TupleVM[ len ];
                for ( var i = 0; i < len; i++ )
                {
                    var w = input_words[ i ];
                    res[ i ]= new ResultVM.TupleVM() { Word = text.Substring( w.startIndex, w.length ), Label = output_words[ i ] };
                }
                return (res);
            }
        }

        public async Task< (List< (IList< ResultVM.TupleVM > result, Exception error) > results, Exception error) > TryRunAsync( string text, string modelType, bool getFirstModelTypeIfMissing = false )
        {
            try
            {
                var mt = GetModelTuple( modelType, getFirstModelTypeIfMissing );
                //------------------------------------------------------------------------------------------------------//

                await _Semaphore.WaitAsync().CAX();
                try
                {
                    var p = GetPredictor( mt );
                    if ( !mt.Tokenizer.TryTokenizeBySentsWithLock( text, out var input_sents ) )
                    {
                        return (EMPTY, EMPTY_Exception);
                    }

                    if ( input_sents.Count == 1 )
                    {
                        try
                        {
                            var input_words  = input_sents[ 0 ];
                            var input_tokens = input_words.Select( w => Tokenizer.ToPosTaggerToken( w, mt.MaxEndingLength ) ).ToList( input_words.Count );
                            var output_words = p.Predict( input_tokens );

                            var res = CreateResultTuples( input_words, output_words, text );
                            return (new List< (IList< ResultVM.TupleVM > result, Exception error) > { (res, default) }, default);
                        }
                        catch ( Exception ex )
                        {
                            return (default, ex);
                        }
                    }
                    else
                    {
                        var sd  = new SortedDictionary< long, (IList< ResultVM.TupleVM > result, Exception error) >();
                        var cnt = 0;
                        var po  = new ParallelOptions() { MaxDegreeOfParallelism = (Environment.ProcessorCount * 4) };
                        Parallel.ForEach( input_sents, po, (input_words, _, i) =>
                        {
                            try
                            {
                                var input_tokens = input_words.Select( w => Tokenizer.ToPosTaggerToken( w, mt.MaxEndingLength ) ).ToList( input_words.Count );
                                var output_words = p.Predict( input_tokens );

                                var res = CreateResultTuples( input_words, output_words, text );

                                sd.AddWithLock( i, (res, default) );
                                Interlocked.Add( ref cnt, res.Count );
                            }
                            catch ( Exception ex )
                            {
                                sd.AddWithLock( i, (default, ex) );
                                Interlocked.Increment( ref cnt );
                            }
                        });
                        var results = sd.Values.ToList( cnt );
                        return (results, default);
                    }
                }
                finally
                {
                    _Semaphore.Release();
                }
            }
            catch ( Exception ex )
            {
                return (default, ex);
            }
        }
        
        
        public IEnumerable< string > GetModelInfoKeys( string regimenModelType ) => regimenModelType.IsNullOrEmpty() ? _SLByType.Keys 
                                                                            : _SLByType.Where( p => string.Compare( p.Value.RegimenModelType, regimenModelType, true ) == 0 ).Select( p => p.Key );

        public async Task LogToFile( string msg )
        {
            if ( _EnableLog )
            {
                await _LogCS.EnterAsync().CAX();
                try
                {
                    await File.AppendAllTextAsync( _LogFileName, $"{DateTime.Now:dd.MM.yyyy, HH:mm}\r\nTEXT: '{msg}'\r\n------------------------------------------------------------------\r\n\r\n", Encoding.UTF8 ).CAX();
                }
                catch ( Exception ex )
                {
                    _EnableLog = false;
                    Debug.WriteLine( ex );
                }
                finally
                {
                    _LogCS.Exit();
                }
            }
        }
        public async Task< string > ReadLogFile()
        {
            await _LogCS.EnterAsync().CAX();
            try
            {
                var text = await File.ReadAllTextAsync( _LogFileName, Encoding.UTF8 ).CAX();
                return (text);
            }
            catch ( Exception ex )
            {
                return (ex.ToString());
            }
            finally
            {
                _LogCS.Exit();
            }
        }
        public async Task DeleteLogFile()
        {
            await _LogCS.EnterAsync().CAX();
            try
            {
                File.Delete( _LogFileName );
            }
            finally
            {
                _LogCS.Exit();
            }
        }
    }
}
