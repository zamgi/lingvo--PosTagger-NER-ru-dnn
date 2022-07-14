using System.Collections.Generic;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Text
{
    /// <summary>
    /// 
    /// </summary>
    public static class BuildInTokens
    {
        public const string EOS = "</s>";
        public const string BOS = "<s>";
        public const string UNK = "<unk>";

        [M(O.AggressiveInlining)] public static bool IsPreDefinedToken( string str )
        {
            switch ( str )
            {
                case EOS: case BOS: case UNK: return (true);
                default: return (false);
            }
        }

        /// <summary>
        /// Pad given sentences to the same length and return their original length
        /// </summary>
        [M(O.AggressiveInlining)] public static float[] PadSentences( List< List< string > > seqs, int maxLen = -1 )
        {
            var originalLengths = new float[ seqs.Count ];
            if ( maxLen <= 0 )
            {
                foreach ( var lst in seqs )
                {
                    if ( maxLen < lst.Count )
                    {
                        maxLen = lst.Count;
                    }
                }
            }

            for ( var i = 0; i < seqs.Count; i++ )
            {
                var s_i = seqs[ i ];
                originalLengths[ i ] = s_i.Count;
                for ( int j = 0, len = maxLen - s_i.Count; j < len; j++ )
                {
                    s_i.Add( EOS );
                }
            }
            return (originalLengths);
        }
        [M(O.AggressiveInlining)] public static void PadSentences_2( List< List< string > > seqs, int maxLen = -1 )
        {
            if ( maxLen <= 0 )
            {
                foreach ( var lst in seqs )
                {
                    if ( maxLen < lst.Count )
                    {
                        maxLen = lst.Count;
                    }
                }
            }

            for ( var i = 0; i < seqs.Count; i++ )
            {
                var s_i = seqs[ i ];
                for ( int j = 0, len = maxLen - s_i.Count; j < len; j++ )
                {
                    s_i.Add( EOS );
                }
            }
        }
    }
}
