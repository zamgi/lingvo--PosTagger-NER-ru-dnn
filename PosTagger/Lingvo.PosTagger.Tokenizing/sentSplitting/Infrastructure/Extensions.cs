using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.SentSplitting
{   
    /// <summary>
    /// 
    /// </summary>
    internal static class Extensions
    {
        public static Set< string > ToSet( this IEnumerable< string > seq, bool toUpperInvariant ) => new Set< string >( seq.Select( d => d.TrimEx( toUpperInvariant ) ).Where( d => !d.IsNullOrEmpty() ) );
        public static int GetItemMaxLength( this ICollection< string > coll ) => ((coll.Count != 0) ? coll.Max( d => d.Length ) : 0);
        public static int GetItemMinLength( this ICollection< string > coll ) => ((coll.Count != 0) ? coll.Min( d => d.Length ) : 0);

        //public static Dictionary< string, T > ToDictionary< T >( this IEnumerable< KeyValuePair< string, T > > seq, bool toUpperInvariant )
        //{
        //    var dict = new Dictionary< string, T >();
        //    foreach ( var pair in seq )
        //    {
        //        var key = pair.Key.TrimEx( toUpperInvariant );
        //        if ( key.IsNullOrEmpty() )
        //            continue;

        //        if ( dict.ContainsKey( key ) )
        //            continue;

        //        dict.Add( key, pair.Value );
        //    }
        //    return (dict);
        //}
        //public static int GetItemMaxKeyLength< T >( this Dictionary< string, T > dict ) => ((dict.Count != 0) ? dict.Max( p => p.Key.Length ) : 0);
        //public static int GetItemMinKeyLength< T >( this Dictionary< string, T > dict ) => ((dict.Count != 0) ? dict.Min( p => p.Key.Length ) : 0);

        [M(O.AggressiveInlining)] public static bool IsNullOrEmpty( this string value ) => string.IsNullOrEmpty( value );
        [M(O.AggressiveInlining)] public static bool IsNullOrWhiteSpace( this string value ) => string.IsNullOrWhiteSpace( value );
        public static string TrimStartDot( this string value ) => value.TrimStart( '.' );
        public static string TrimEndDot( this string value ) => value.TrimEnd( '.' );
        unsafe public static string AsString( this IEnumerable< char > chars, int len )
        {
            const int MAX_STACK_LEN = 0x1000;

            if ( MAX_STACK_LEN <= len )
            {
                var array = stackalloc char[ len ];
                var i = 0;
                foreach ( var ch in chars )
                {
                    array[ i++ ] = ch;
                }
                return (new string( array, 0, len ));
            }
            else
            {
                var array = new char[ len ];
                var i = 0;
                foreach ( var ch in chars )
                {
                    array[ i++ ] = ch;
                }
                return (new string( array, 0, len ));
            }
        }

        public static bool AttrValueIsTrue( this XElement xe, string attrName )
        {
            var xa = xe.Attribute( attrName );
            if ( (xa != null) && bool.TryParse( xa.Value, out var r ) )
            {
                return (r);
            }
            return (false);
        }

        private static string TrimEx( this string value, bool toUpperInvarian ) => (toUpperInvarian ? value?.ToUpperInvariant() : value);

        private const char DOT = '.';
        private static readonly char[] SPLIT_BY_DOT    = new[] { DOT };
        private static readonly char[] SPLIT_BY_SPACES = new[] { ' ', '\t', '\r', '\n' };
        public static ngram_t< before_no_proper_t > ToBeforeNoProper_ngrams( this XElement xe )
        {
            var words = xe.GetWordsArray();
            var unstick_from_digits = xe.AttrValueIsTrue( "unstick-from-digits" );
            var digits_after        = xe.AttrValueIsTrue( "digits-after" );

            var ngram = new ngram_t< before_no_proper_t >( words, new before_no_proper_t( unstick_from_digits, digits_after ) );
            return (ngram);
        }
        public static ngram_t< before_proper_or_number_t > ToBeforeProperOrNumber_ngrams( this XElement xe )
        {
            var words = xe.GetWordsArray();
            var digits_before       = xe.AttrValueIsTrue( "digits-before" );
            var slash_before        = xe.AttrValueIsTrue( "slash-before" );
            var unstick_from_digits = xe.AttrValueIsTrue( "unstick-from-digits" );

            var ngram = new ngram_t< before_proper_or_number_t >( words, new before_proper_or_number_t( digits_before, slash_before, unstick_from_digits ) );
            return (ngram);
        }

        private static string[] GetWordsArray( this XElement xe )
        {
            var words = xe.Value.Split( SPLIT_BY_DOT, StringSplitOptions.RemoveEmptyEntries );
            var word_list = new List< string >( words.Length );
            for ( int i = 0, len = words.Length - 1; i <= len; i++ )
            {
                var word = words[ i ].Trim();
                var words_by_space = word.Split( SPLIT_BY_SPACES, StringSplitOptions.RemoveEmptyEntries );
                if ( words_by_space.Length == 1 )
                {
                    //if ( i == len )
                    //{
                    //    word_list.Add( word );
                    //}
                    //else
                    //{
                        word_list.Add( word + DOT );
                    //}                    
                }
                else
                {
                    for ( int j = 0, len_by_space = words_by_space.Length - 1; j <= len_by_space; j++ )
                    {
                        word = words_by_space[ j ];
                        if ( j == len_by_space )
                        {
                            //if ( i == len )
                            //{
                            //    word_list.Add( word );
                            //}
                            //else
                            //{
                                word_list.Add( word + DOT );
                            //} 
                        }
                        else
                        {
                            word_list.Add( word );
                        }  
                    }
                }
            }
            return (word_list.ToArray());
        }
    }
}
