using System.Collections.Generic;
using System.Linq;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Metrics;
using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Text;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    public static class Validator
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly struct MetricInfo
        {
            public string MetricName { get; init; }
            public double Score      { get; init; }
            public string Text       { get; init; }
            public override string ToString() => $"{MetricName}, Score={Score:F4}, {Text}";
        }
        /// <summary>
        /// 
        /// </summary>
        public readonly struct Result
        {
            public IReadOnlyList< MetricInfo > MetricInfos { get; init; }
            public override string ToString() => (MetricInfos.AnyEx() ? string.Join("; ", MetricInfos ) : "-");
        }

        public static Result Run_Validate( Options opts )
        {
            Logger.WriteLine( $"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPath}'" );

            using var cts = Console_CancelKeyPress_Breaker.Create();

            // Load valid corpus
            using var validCorpus = new Corpus( opts.ValidCorpusPath, opts.BatchSize, opts.MaxPredictSentLength, opts.TooLongSequence, cts.Token );

            // Load model
            var sl = SeqLabel.Create4Predict( opts );

            // Create metrics
            //---var metrics = CreateFromTgtVocab_MultiLabelsFscoreMetric( validCorpus );
            var metrics = CreateFromTgtVocab_MultiLabelsFscoreMetric( sl.TgtVocab );

            sl.Validate( validCorpus, metrics, cts.Token );

            // throw if canceled
            cts.Cts.Token.ThrowIfCancellationRequested();

            var mis = (from m in metrics
                          let t = m.GetScore()
                          select new MetricInfo()
                          {
                             MetricName = m.Name,
                             Score      = t.primaryScore,
                             Text       = t.text
                          }
                         ).ToList();
            return (new Result() { MetricInfos = mis });
        }

        public static IList< IMetric > CreateFromTgtVocab_SeqLabelFscoreMetric( Corpus corpus, int vocabSize, bool vocabIgnoreCase = false )
        {
            var tgtVocab = corpus.BuildTargetVocab( vocabIgnoreCase, vocabSize );
            return (CreateFromTgtVocab_SeqLabelFscoreMetric( tgtVocab ));
        }
        public static IList< IMetric > CreateFromTgtVocab_MultiLabelsFscoreMetric( Corpus corpus, int vocabSize, bool vocabIgnoreCase = false )
        {
            var tgtVocab = corpus.BuildTargetVocab( vocabIgnoreCase, vocabSize );
            return (CreateFromTgtVocab_MultiLabelsFscoreMetric( tgtVocab ));
        }
        public static IList< IMetric > CreateFromTgtVocab_SeqLabelFscoreMetric( Vocab tgtVocab )
        {
            var metrics = (from word in tgtVocab.Items.OrderBy( TagToOrderPos )
                           where !BuildInTokens.IsPreDefinedToken( word )
                           select (IMetric) new SeqLabelFscoreMetric( word )
                          ).ToList( tgtVocab.Items.Count );
            return (metrics);
        }
        public static IList< IMetric > CreateFromTgtVocab_MultiLabelsFscoreMetric( Vocab tgtVocab )
        {
            var labels = from word in tgtVocab.Items.OrderBy( TagToOrderPos )
                         where !BuildInTokens.IsPreDefinedToken( word ) && (word != "O")
                         select word;
            var metrics = new[]
            {
                new MultiLabelsFscoreMetric( labels, "all" ),
            };
            return (metrics);
        }

        public static string TagToOrderPos( string tag ) => tag;
    }
}

