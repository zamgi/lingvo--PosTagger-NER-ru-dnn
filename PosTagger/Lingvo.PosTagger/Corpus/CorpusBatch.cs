using System.Collections.Generic;
using System.Linq;

using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Text
{
    /// <summary>
    /// 
    /// </summary>
    public class CorpusBatch
    {
        protected List< List< string > > _SrcTokensBatch; // shape: (batch_size, seq_size)
        protected List< List< string > > _TgtTokensBatch; // shape: (batch_size, seq_size)
        protected List< SntPair > _SntPairs;
        private int _SrcTotalTokensCount;
        private int _TgtTotalTokensCount;

        private CorpusBatch( List< List< string > > srcTokensBatch, List< List< string > > tgtTokensBatch ) 
        {
            _SrcTokensBatch = srcTokensBatch;
            _TgtTokensBatch = tgtTokensBatch;

            _SrcTotalTokensCount = srcTokensBatch.Sum( tokens => tokens.Count );
            _TgtTotalTokensCount = tgtTokensBatch.Sum( tokens => tokens.Count );
        }
        public CorpusBatch( List< string > srcTokens )
        {
            _SrcTokensBatch = new List< List< string > > { srcTokens };
            _TgtTokensBatch = CreateTgtTokensBatch( batchSize: 1, srcTokens.Count );

            _SrcTotalTokensCount = _TgtTotalTokensCount = srcTokens.Count;            
        }
        public CorpusBatch( List< SntPair > sntPairs ) => CreateBatch( sntPairs );

        private void CreateBatch( List< SntPair > sntPairs )
        {
            _SntPairs = sntPairs;

            var batchSize = sntPairs.Count;
            _SrcTokensBatch = new List< List< string > >( batchSize );
            _TgtTokensBatch = new List< List< string > >( batchSize );

            _SrcTotalTokensCount = 0;
            _TgtTotalTokensCount = 0;
            for ( var i = 0; i < batchSize; i++ )
            {
                var sp = sntPairs[ i ];

                _SrcTokensBatch.Add( sp.SrcTokens );
                _SrcTotalTokensCount += sp.SrcTokens.Count;

                _TgtTokensBatch.Add( sp.TgtTokens );
                _TgtTotalTokensCount += sp.TgtTokens.Count;
            }
        }
        private static List< List< string > > CreateTgtTokensBatch( int batchSize, int tgtTokenCapacity )
        {
            var tgtTokensBatch = new List< List< string > >( batchSize );
            for ( int i = 0; i < batchSize; i++ )
            {
                tgtTokensBatch.Add( new List< string >( tgtTokenCapacity ) );
            }
            return (tgtTokensBatch);
        }

        public IReadOnlyList< SntPair > SntPairs => _SntPairs;
        public int SrcTotalTokensCount => _SrcTotalTokensCount;
        public int TgtTotalTokensCount => _TgtTotalTokensCount;

        public List< List< string > > SrcTokensBatch => _SrcTokensBatch;
        public List< List< string > > TgtTokensBatch => _TgtTokensBatch;

        public int GetBatchSize() => _SrcTokensBatch.Count;

        public CorpusBatch GetRange( int idx, int count ) => new CorpusBatch( _SrcTokensBatch.GetRange( idx, count ), _TgtTokensBatch.GetRange( idx, count ) );
        public CorpusBatch CloneSrcTokens() => new CorpusBatch( _SrcTokensBatch, CreateTgtTokensBatch( this.GetBatchSize(), _TgtTotalTokensCount ) );
    }

    /// <summary>
    /// 
    /// </summary>
    public sealed class CorpusBatchBuilder
    {
        private Dictionary< string, int > _Src_dict;
        private Dictionary< string, int > _Tgt_dict;
        public CorpusBatchBuilder()
        {
            _Src_dict = new Dictionary< string, int >();
            _Tgt_dict = new Dictionary< string, int >();
        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// side and the second group in target side are shared vocabulary
        public void CountSntPairTokens( IEnumerable< SntPair > sntPairs )
        {
            foreach ( var sp in sntPairs )
            {
                var src_tokens = sp.SrcTokens;
                for ( int i = 0, len = src_tokens.Count; i < len; i++ )
                {
                    var token = src_tokens[ i ];
                    if ( _Src_dict.TryGetValue( token, out var cnt ) )
                    {
                        _Src_dict[ token ] = cnt + 1;
                    }
                    else
                    {
                        _Src_dict.Add( token, 1 );
                    }
                }

                var tgt_tokens = sp.TgtTokens;
                for ( int i = 0, len = tgt_tokens.Count; i < len; i++ )
                {
                    var token = tgt_tokens[ i ];
                    if ( _Tgt_dict.TryGetValue( token, out var cnt ) )
                    {
                        _Tgt_dict[ token ] = cnt + 1;
                    }
                    else
                    {
                        _Tgt_dict.Add( token, 1 );
                    }
                }
            }
        }
        public void CountTargetTokens( IEnumerable< SntPair > sntPairs )
        {
            foreach ( var sp in sntPairs )
            {
                var tgt_tokens = sp.TgtTokens;
                for ( int i = 0, len = tgt_tokens.Count; i < len; i++ )
                {
                    var token = tgt_tokens[ i ];
                    if ( _Tgt_dict.TryGetValue( token, out var cnt ) )
                    {
                        _Tgt_dict[ token ] = cnt + 1;
                    }
                    else
                    {
                        _Tgt_dict.Add( token, 1 );
                    }
                }
            }
        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="sharedSrcTgtVocabGroupMapping">The mappings for shared vocabularies between source side and target side. The values in the mappings are group ids. For example: sharedSrcTgtVocabGroupMapping[0] = 1 means the first group in source
        /// side and the second group in target side are shared vocabulary</param>
        public (Vocab srcVocab, Vocab tgtVocab) GenerateVocabs( bool vocabIgnoreCase, int vocabSize ) => GenerateVocabs( vocabIgnoreCase, vocabIgnoreCase, vocabSize, vocabSize );
        public (Vocab srcVocab, Vocab tgtVocab) GenerateVocabs( bool srcVocabIgnoreCase, bool tgtVocabIgnoreCase, int srcVocabSize, int tgtVocabSize )
        {
            //---Logger.WriteLine( $"Building vocabulary from corpus." );

            var srcVocab = InnerBuildVocab( srcVocabIgnoreCase, srcVocabSize, _Src_dict, "Source" );
            var tgtVocab = InnerBuildVocab( tgtVocabIgnoreCase, tgtVocabSize, _Tgt_dict, "Target" );

            return (srcVocab, tgtVocab);
        }
        public Vocab GenerateTargetVocab( bool vocabIgnoreCase, int vocabSize ) => InnerBuildVocab( vocabIgnoreCase, vocabSize, _Tgt_dict, "Target" );

        /// <summary>
        /// 
        /// </summary>
        private sealed class DescendingComparer : IComparer< int >
        {
            public int Compare( int x, int y ) => y.CompareTo( x );
        }
        private static Vocab InnerBuildVocab( bool ignoreCase, int vocabSize, Dictionary< string, int > d, string vocabType )
        {
            var sd = new SortedDictionary< int, List< string > >( new DescendingComparer() );
            
            foreach ( var (token, freq) in d )
            {
                if ( BuildInTokens.IsPreDefinedToken( token ) ) continue;

                if ( !sd.TryGetValue( freq, out var lst ) )
                {
                    lst = new List< string >();
                    sd.Add( freq, lst );
                }
                lst.Add( token );
            }

            var v = Vocab.CreateDicts( ignoreCase: ignoreCase, d.Count );
            var wordToIndex = v.wordToIndex;
            var indexToWord = v.indexToWord;
            var q = Vocab.START_MEANING_INDEX;

            Vocab finita()
            {
                Logger.WriteLine( $"{vocabType}: Original vocabulary size = '{d.Count:#,#}', Truncated (with pre-defined token) vocabulary size = '{q:#,#}'" );

                var vocab = new Vocab( v );
                return (vocab);
            };

            foreach ( var tokens in sd.Values )
            {
                foreach ( var token in tokens )
                {
                    // add word to vocab
                    wordToIndex[ token ] = q;
                    indexToWord[ q     ] = token;
                    q++;

                    if ( vocabSize <= q )
                    {
                        return (finita());
                    }
                }
            }

            return (finita());
        }
    }
}
