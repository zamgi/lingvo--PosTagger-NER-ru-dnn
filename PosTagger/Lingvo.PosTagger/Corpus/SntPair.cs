using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

using Lingvo.PosTagger.Utils;
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Text
{
    /// <summary>
    /// 
    /// </summary>
    public struct /*class*/ RawSntPair
    {
        private string _SrcSnt;
        private string _TgtSnt;

        private long _SrcGroupLenId;
        private long _TgtGroupLenId;
        private long _GroupLenId;

        private int _SrcLength;
        private int _TgtLength;

        private long _MaxSeqLength;
        public RawSntPair( string srcSnt, string tgtSnt, int maxSrcLength, int maxTgtLength, bool truncateTooLongSeq )
        {
            _MaxSeqLength = Math.Max( maxSrcLength, maxTgtLength );

            _SrcLength = CountWhiteSpace( srcSnt );
            _TgtLength = CountWhiteSpace( tgtSnt );

            if ( truncateTooLongSeq )
            {
                if ( maxSrcLength < _SrcLength )
                {
                    srcSnt = Truncate( srcSnt, maxSrcLength );
                    _SrcLength = CountWhiteSpace( srcSnt );
                }
                if ( maxTgtLength < _TgtLength )
                {
                    tgtSnt = Truncate( tgtSnt, maxTgtLength );
                    _TgtLength = CountWhiteSpace( tgtSnt );
                }
            }

            _SrcGroupLenId = _SrcLength; // CountWhiteSpace( srcSnt ) == GenerateGroupLenId( srcSnt ); //_SrcGroupLenId = GenerateGroupLenId( srcSnt, _MaxSeqLength );
            _TgtGroupLenId = _TgtLength; // CountWhiteSpace( tgtSnt ) == GenerateGroupLenId( tgtSnt ); //_TgtGroupLenId = GenerateGroupLenId( tgtSnt, _MaxSeqLength );
            _GroupLenId    = GenerateGroupLenId( srcSnt, tgtSnt, _MaxSeqLength );                      //_GroupLenId    = GenerateGroupLenId( srcSnt + '\t' + tgtSnt, _MaxSeqLength );

            _SrcSnt = srcSnt;
            _TgtSnt = tgtSnt;
        }

        public string SrcSnt => _SrcSnt;
        public string TgtSnt => _TgtSnt;
        public long SrcGroupLenId => _SrcGroupLenId;
        public long TgtGroupLenId => _TgtGroupLenId;
        public long GroupLenId => _GroupLenId;
        public int SrcLength => _SrcLength;
        public int TgtLength => _TgtLength;

        //[M(O.AggressiveInlining)] private static long GenerateGroupLenId__PREV( string s, long maxSeqLength )
        //{
        //    long r = 0;
        //    var array = s.Split( '\t' );
        //    foreach ( var a in array )
        //    {
        //        r *= maxSeqLength;
        //        r += CountWhiteSpace( a ); //---r += = a.Split( ' ' ).Length;
        //    }
        //    return (r);
        //}
        //[M(O.AggressiveInlining)] private static long GenerateGroupLenId( string src_or_tgt ) => CountWhiteSpace( src_or_tgt );
        [M(O.AggressiveInlining)] private static long GenerateGroupLenId( string src, string tgt, long maxSeqLength ) => CountWhiteSpace( src ) * maxSeqLength + CountWhiteSpace( tgt );        
        [M(O.AggressiveInlining)] internal static int CountWhiteSpace( string s )
        {
            var cnt = 0;
            for ( var i = s.Length - 1; 0 <= i; i-- )
            {
                if ( s[ i ] == ' ' )
                {
                    cnt++;
                }
            }
            return (cnt);
        }
        //[M(O.AggressiveInlining)] private static string Truncate__PREV( string txt, int maxSeqLength )
        //{
        //    var array   = txt.Split( '\t' );
        //    var results = new List< string >( array.Length );

        //    foreach ( var a in array )
        //    {
        //        var tokens = a.Split( ' ' );
        //        if ( tokens.Length <= maxSeqLength )
        //        {
        //            results.Add( a );
        //        }
        //        else
        //        {
        //            results.Add( string.Join( ' ', tokens, 0, maxSeqLength ) );
        //        }
        //    }

        //    return (string.Join( "\t", results ));
        //}
        [M(O.AggressiveInlining)] private static string Truncate( string src_or_tgt, int maxLength )
        {
            var tokens = src_or_tgt.Split( ' ' );
            if ( tokens.Length <= maxLength )
            {
                return (src_or_tgt);
            }

            var truncated__src_or_tgt = string.Join( ' ', tokens, 0, maxLength );
            return (truncated__src_or_tgt);
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public struct/*class*/ SntPair
    {
        private List< string > _SrcTokens;
        private List< string > _TgtTokens;

        public SntPair( string src_line, string tgt_line )
        {
            _SrcTokens = CreateTokens( src_line );
            _TgtTokens = CreateTokens( tgt_line );
        }

        public List< string > SrcTokens { [M(O.AggressiveInlining)] get => _SrcTokens; }
        public List< string > TgtTokens { [M(O.AggressiveInlining)] get => _TgtTokens; }

        public string PrintSrcTokens() => string.Join( " ", _SrcTokens );
        public string PrintTgtTokens() => string.Join( " ", _TgtTokens );

        [M(O.AggressiveInlining)] private static List< string > CreateTokens( string line )
        {
            var tokens = TokenizeBySpace( line );
//#if DEBUG
//            Debug.Assert( tokens.SequenceEqual( line.Split( ' ' ) ) );
//#endif
            return (tokens);

            //var array = line.Split( ' ' );
            //var res = array.ToList( array.Length );
            //return (res);
        }
        [M(O.AggressiveInlining)] private static List< string > TokenizeBySpace( string line )
        {
            var cnt = RawSntPair.CountWhiteSpace( line );
            var lst = new List< string >( cnt + 1 );
            if ( 0 < cnt )
            {
                var si = 0;
                for ( var i = line.IndexOf( ' ' ); i != -1; )
                {
                    var token = line.Substring( si, i - si );
                    lst.Add( token );
                    si = i + 1;
                    i = line.IndexOf( ' ', si );
                }

                var token_2 = line.Substring( si );
                lst.Add( token_2 );
            }
            else
            {
                lst.Add( line );
            }
            return (lst);
        }
    }
}
