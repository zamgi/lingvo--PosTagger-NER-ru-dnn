using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    public static partial class PosTaggerExtensions
    {
        public static void SetPosTaggerOutputType( this IList< word_t > words, IList< string > output_words, bool correctByInputType = true, bool setSeqLabelOutputType = true )
        {
            //---Debug.Assert( words.Count == output_words.Count );

            var i = Math.Min( words.Count, output_words.Count ) - 1;
            if ( correctByInputType )
            {
                for ( ; 0 <= i; i-- )
                {
                    var w  = words[ i ];
                    var ow = output_words[ i ];
                    switch ( w.posTaggerInputType )
                    {
                        case PosTaggerInputType.Num: w.posTaggerOutputType = PosTaggerOutputType.Numeral; break;
                        //case PosTaggerInputType.Email: w.posTaggerOutputType = PosTaggerOutputType.Email;   break;
                        //case PosTaggerInputType.Url  : w.posTaggerOutputType = PosTaggerOutputType.Url;     break;
                        case PosTaggerInputType.Email: 
                        case PosTaggerInputType.Url  : w.posTaggerOutputType = PosTaggerOutputType.Other; break;

                        default:
                            if ( (w.extraWordType & ExtraWordType.Punctuation) == ExtraWordType.Punctuation )
                            {
                                w.posTaggerOutputType = PosTaggerOutputType.Punctuation;
                            }
                            else
                            {
                                w.posTaggerOutputType = ow.ToPosTaggerOutputType();                                
                            }                            
                            break;
                    }
                    if ( setSeqLabelOutputType ) w.seqLabelOutputType = ow;
                }
            }
            else
            {
                for ( ; 0 <= i; i-- )
                {
                    var w  = words[ i ];
                    var ow = output_words[ i ];
                    w.posTaggerOutputType = ow.ToPosTaggerOutputType();
                    if ( setSeqLabelOutputType ) w.seqLabelOutputType  = ow;
                }
            }
        }

        [M(O.AggressiveInlining)] public static bool TryTokenizeBySents( this Tokenizer tokenizer, string text, out IList< List< word_t > > input_sents )
        {
            var sents = tokenizer.Run_SimpleSentsAllocate( text ); //--var sents = tokenizer.Run( text ); //---
            if ( 0 < sents.Count )
            {
                input_sents = sents.Where( s => 0 < s.Count ).ToList( sents.Count );
                return (0 < input_sents.Count);
            }

            input_sents = default;
            return (false);
        }
        [M(O.AggressiveInlining)] public static bool TryTokenizeBySentsWithLock( this Tokenizer tokenizer, string text, out IList< List< word_t > > input_sents )
        {
            lock ( tokenizer )
            {
                return (tokenizer.TryTokenizeBySents( text, out input_sents ));
            }
        }
        [M(O.AggressiveInlining)] public static  IEnumerable< word_t > Run_SimpleSentsAllocate_Unbend( this Tokenizer tokenizer, string text )
        {
            var sents = tokenizer.Run_SimpleSentsAllocate( text );
            return (sents.SelectMany( s => s ));
        }
        [M(O.AggressiveInlining)] public static List< word_t > Run_SimpleSentsAllocate_UnbendList( this Tokenizer tokenizer, string text )
        {
            var sents = tokenizer.Run_SimpleSentsAllocate( text );
            if ( sents.Count == 1 )
            {
                return (sents[ 0 ]);
            }
            return (sents.SelectMany( s => s ).ToList( sents.Sum( s => s.Count ) ));
        }
        [M(O.AggressiveInlining)] internal static List< T > ToList< T >( this IEnumerable< T > seq, int capatity )
        {
            var lst = new List< T >( capatity );
            lst.AddRange( seq );
            return (lst);
        }
    }
}