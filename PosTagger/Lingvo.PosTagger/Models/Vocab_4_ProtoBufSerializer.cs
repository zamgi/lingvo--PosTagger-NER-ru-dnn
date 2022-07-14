using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Lingvo.PosTagger.Text;
using Lingvo.PosTagger.Utils;

using ProtoBuf;

namespace Lingvo.PosTagger.Models
{
    /// <summary>
    /// 
    /// </summary>
    [ProtoContract(SkipConstructor=true)]
    public sealed class Vocab_4_ProtoBufSerializer
    {
        public const int START_MEANING_INDEX = 3;

        [ProtoMember(1)] private Dictionary< string, int > _WordToIndex;
        [ProtoMember(2)] private Dictionary< int, string > _IndexToWord;
        [ProtoMember(3)] private bool _IgnoreCase;
        public int Count => _IndexToWord.Count;
        public bool IgnoreCase => _IgnoreCase;
        public IReadOnlyCollection< string > Items => _WordToIndex.Keys;
        public Dictionary< string, int > _GetWordToIndex_() => _WordToIndex;
        public Dictionary< int, string > _GetIndexToWord_() => _IndexToWord;
        public Vocab ToVocab() => new Vocab( this );

        public static (Dictionary< string, int > wordToIndex, Dictionary< int, string > indexToWord, bool ignoreCase) CreateDicts( bool ignoreCase )
        {
            var wordToIndex = ignoreCase ? new Dictionary< string, int >( StringComparer.InvariantCultureIgnoreCase ) 
                                         : new Dictionary< string, int >();
            var indexToWord = new Dictionary< int, string >();

            wordToIndex[ BuildInTokens.EOS ] = (int) SentTagsEnum.END;
            wordToIndex[ BuildInTokens.BOS ] = (int) SentTagsEnum.START;
            wordToIndex[ BuildInTokens.UNK ] = (int) SentTagsEnum.UNK;

            indexToWord[ (int) SentTagsEnum.END   ] = BuildInTokens.EOS;
            indexToWord[ (int) SentTagsEnum.START ] = BuildInTokens.BOS;
            indexToWord[ (int) SentTagsEnum.UNK   ] = BuildInTokens.UNK;

            return (wordToIndex, indexToWord, ignoreCase);
        }

        public Vocab_4_ProtoBufSerializer( Vocab v ) => (_WordToIndex, _IndexToWord, _IgnoreCase) = (v._GetWordToIndex_(), v._GetIndexToWord_(), v.IgnoreCase);
        public Vocab_4_ProtoBufSerializer( bool ignoreCase ) => (_WordToIndex, _IndexToWord, _IgnoreCase) = CreateDicts( ignoreCase );
        public Vocab_4_ProtoBufSerializer( Dictionary< string, int > wordToIndex, Dictionary< int, string > indexToWord ) => (_WordToIndex, _IndexToWord) = (wordToIndex, indexToWord);
        public Vocab_4_ProtoBufSerializer( in (Dictionary< string, int > wordToIndex, Dictionary< int, string > indexToWord) t ) => (_WordToIndex, _IndexToWord) = t;
        /// <summary>
        /// Load vocabulary from given files
        /// </summary>
        public Vocab_4_ProtoBufSerializer( string vocabFilePath, bool ignoreCase )
        {
            Logger.WriteLine( "Loading vocabulary files..." );

            (_WordToIndex, _IndexToWord, _IgnoreCase) = CreateDicts( ignoreCase );

            using var sr = new StreamReader( vocabFilePath );
            //Build word index for both source and target sides
            int q = START_MEANING_INDEX;
            for ( var line = sr.ReadLine(); line != null; line = sr.ReadLine() )
            {
                var idx = line.IndexOf( '\t' );
                var word = (idx == -1) ? line : line.Substring( 0, idx );
                if ( word.IsNullOrEmpty() ) continue;

                if ( !BuildInTokens.IsPreDefinedToken( word ) )
                {
                    _WordToIndex[ word ] = q;
                    _IndexToWord[ q ] = word;
                    q++;
                }
            }
        }

        public void MergeWith( Vocab_4_ProtoBufSerializer vocab )
        {
            var maxId = 0;
            foreach ( var id in _WordToIndex.Values )
            {
                if ( maxId < id )
                {
                    maxId = id;
                }
            }

            maxId++;
            foreach ( var p in vocab._WordToIndex )
            {
                if ( !_WordToIndex.ContainsKey( p.Key ) )
                {
                    _WordToIndex.Add( p.Key, maxId );
                    _IndexToWord.Add( maxId, p.Key );
                    maxId++;
                }
            }
        }
        public void Dump( string fileName )
        {
            using var sw = new StreamWriter( fileName, append: false, Encoding.UTF8 );
            foreach ( var p in _IndexToWord )
            {
                sw.Write( p.Value );
                sw.Write( '\t' );
                sw.WriteLine( p.Key );
            }
        }

        public string GetWordByIndex( int idx )
        {
            if ( _IndexToWord.TryGetValue( idx, out var letter ) )
            {
                return (letter);
            }
            return (BuildInTokens.UNK);
        }
        public List< string > GetWordsByIndices( List< float > idxs )
        {
            var result = new List< string >( idxs.Count );
            foreach ( int idx in idxs )
            {
                if ( !_IndexToWord.TryGetValue( idx, out var letter ) )
                {
                    letter = BuildInTokens.UNK;
                }
                result.Add( letter );
            }
            return (result);
        }
        public string GetIndexByWord( int idx )
        {
            if ( !_IndexToWord.TryGetValue( idx, out var letter ) )
            {
                letter = BuildInTokens.UNK;
            }
            return (letter);
        }
        public bool ContainsWord( string word ) => _WordToIndex.ContainsKey( word );
        public List< List< int > > GetIndicesByWords( List< List< string > > seqs )
        {
            var result = new List< List< int > >( seqs.Count );
            foreach ( var seq in seqs )
            {
                var r = new List< int >( seq.Count );
                foreach ( var word in seq )
                {
                    if ( !_WordToIndex.TryGetValue( word, out int id ) )
                    {
                        id = (int) SentTagsEnum.UNK;
                    }
                    r.Add( id );
                }
                result.Add( r );
            }
            return (result);
        }
    }
}
