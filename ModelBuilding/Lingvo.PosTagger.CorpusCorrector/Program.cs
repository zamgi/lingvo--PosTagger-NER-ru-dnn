using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

using Lingvo.PosTagger.Tokenizing;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    internal static class Program
    {
        private static string SENT_SPLITTER_RESOURCES => "../../../[resources]/tokenizing/sent-splitter-resources.xml";
        private static string URL_DETECTOR_RESOURCES  => "../../../[resources]/tokenizing/url-detector-resources.xml";

        private static void Main()
        {
            try
            {
                const string TRAIN_FOLDER = "../../../[resources]/train/custom_corpus_1/";
                const string VALID_FOLDER = "../../../[resources]/valid/custom_corpus_1/";

                #region comm.
                /*
                var tokenizerConfig = new TokenizerConfig( SENT_SPLITTER_RESOURCES, URL_DETECTOR_RESOURCES );
                using var tokenizer = new Tokenizer( tokenizerConfig, replaceNumsOnPlaceholder: true );
                var words_1 = tokenizer.Run_SimpleSentsAllocate( "Варкалось.Хливкие шорьки " );
                var words_2 = tokenizer.Run( "Варкалось.Хливкие шорьки " );
                
                var words_3 = tokenizer.Run_SimpleSentsAllocate( "стоимостью 18. 3 млн руб." );
                var words_4 = tokenizer.Run( "стоимостью 18. 3 млн руб." );
                //*/
                #endregion
                //---------------------------------------------------------------------//

                #region [.1.]
                //clear( TRAIN_FOLDER + "___corpus_pos_tagger_ru_1.txt" );
                #endregion

                #region [.2.]
                //var tokenizerConfig = new TokenizerConfig( SENT_SPLITTER_RESOURCES, URL_DETECTOR_RESOURCES );
                //ReplaceNumsOnPlaceholder( tokenizerConfig,
                //                          TRAIN_FOLDER + "___corpus_pos_tagger_ru_1.txt",
                //                          TRAIN_FOLDER + "___corpus_pos_tagger_ru_2.txt",
                //                          replaceNumsOnPlaceholder: true );
                #endregion

                #region [.3.]
                static void func( int maxEndingLength, float validPercent )
                {
                    var outfile = TRAIN_FOLDER + $"__corpus_pos_tagger_ru__(mel={maxEndingLength})__full.txt";
                    conv_1( TRAIN_FOLDER + "___corpus_pos_tagger_ru_2.txt", outfile, maxEndingLength );

                    split_by_train_and_valid( outfile,
                                              TRAIN_FOLDER + $"train_pos_tagger_ru__(mel={maxEndingLength}).txt",
                                              VALID_FOLDER + $"valid_pos_tagger_ru__(mel={maxEndingLength}).txt",
                                              validPercent
                                            );
                };

                func( maxEndingLength: 4, validPercent: 5.0f );
                func( maxEndingLength: 5, validPercent: 5.0f );
                //func( maxEndingLength: 7, validPercent: 5.0f );
                //func( maxEndingLength: 9, validPercent: 5.0f );
                #endregion
            }
            catch ( Exception ex )
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine( Environment.NewLine + ex + Environment.NewLine );
                Console.ResetColor();
            }
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine( "\r\n\r\n  [.......finita.......]" );
            Console.ReadLine();
        }

        private static void conv_1( string inputFileName, string outputFileName, int maxEndingLength )
        {
            Console.WriteLine( $"\r\n-------------------------------------------------\r\n file: '{inputFileName}' => '{outputFileName}'" );

            using var sr = new StreamReader( inputFileName );
            using var sw = new StreamWriter( outputFileName, append: false, Encoding.UTF8 );

            var sep = new[] { ' ' };
            for ( var line = sr.ReadLine(); line != null; line = sr.ReadLine() )
            {
                if ( line.IsNullOrWhiteSpace() )
                {
                    sw.WriteLine();
                    continue;
                }

                var idx = line.IndexOf( '\t' );
                Debug.Assert( idx != -1 );
                var token = line.Substring( 0, idx  );
                var pos   = line.Substring( idx + 1 );

                var new_token = Tokenizer.ToPosTaggerToken( token, maxEndingLength );
                
                idx = new_token.IndexOf( ' ' );
                if ( idx != -1 )
                {
                    var array = new_token.Split( sep, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries );
                    foreach ( var a in array )
                    {
                        sw.Write( a );
                        sw.Write( '\t' );
                        sw.WriteLine( pos );
                    }
                }
                else
                {
                    sw.Write( new_token );
                    sw.Write( '\t' );
                    sw.WriteLine( pos );
                }
            }
        }
        private static void split_by_train_and_valid( string inputFileName, string outputTrainFileName, string outputValidFileName, float validPercent = 5.0f )
        {
            Console.WriteLine( $"\r\n-------------------------------------------------\r\n file: '{inputFileName}'" );

            using var sr = new StreamReader( inputFileName );

            var sent = new List< (string token, string pos) >();
            var all_sents = new List< IList< (string token, string pos) > >();

            for ( var line = sr.ReadLine(); line != null; line = sr.ReadLine() )
            {
                if ( !line.IsNullOrWhiteSpace() )
                {
                    var idx = line.IndexOf( '\t' );
                    Debug.Assert( idx != -1 );
                    var token = line.Substring( 0, idx  );
                    var pos   = line.Substring( idx + 1 );

                    sent.Add( (token, pos) ); 
                    continue; 
                }

                //---Debug.Assert( 1 < sent.Count );
                switch ( sent.Count )
                {
                    case 0: break;
                    case 1: case 2:
                        Console.WriteLine( $"'{string.Join( "', '", sent )}'" );
                        break;
                    default:
                        all_sents.Add( sent.ToArray() );
                        sent.Clear();
                        break;
                }
            }

            using var valid_sw = new StreamWriter( outputValidFileName, append: false, Encoding.UTF8 );

            var validCount = (int) ((validPercent * all_sents.Count) / 100);
            var rnd = new Random();
            for ( ; 0 < validCount; validCount-- )
            {
                var i = rnd.Next( 0, all_sents.Count );
                var valid_sent = all_sents[ i ];
                all_sents.RemoveAt( i );

                foreach ( var (token, pos) in valid_sent )
                {
                    valid_sw.Write( token );
                    valid_sw.Write( '\t' );
                    valid_sw.WriteLine( pos );
                }
                valid_sw.WriteLine();
            }

            using var train_sw = new StreamWriter( outputTrainFileName, append: false, Encoding.UTF8 );

            foreach ( var train_sent in all_sents )
            {
                foreach ( var (token, pos) in train_sent )
                {
                    train_sw.Write( token );
                    train_sw.Write( '\t' );
                    train_sw.WriteLine( pos );
                }
                train_sw.WriteLine();
            }
        }
        private static void clear( string inputFileName )
        {
            Console.WriteLine( $"\r\n-------------------------------------------------\r\n file: '{inputFileName}'" );

            using var sr = new StreamReader( inputFileName );

            var sent = new List< (string token, string pos) >();
            var all_sents = new List< IList< (string token, string pos) > >();

            for ( var line = sr.ReadLine(); line != null; line = sr.ReadLine() )
            {
                if ( !line.IsNullOrWhiteSpace() )
                {
                    var idx = line.IndexOf( '\t' );
                    Debug.Assert( idx != -1 );
                    var token = line.Substring( 0, idx  );
                    var pos   = line.Substring( idx + 1 );

                    sent.Add( (token, pos) ); 
                    continue; 
                }

                //---Debug.Assert( 1 < sent.Count );
                switch ( sent.Count )
                {
                    case 0: break;
                    case 1: case 2:
                        Console.WriteLine( $"'{string.Join( "', '", sent )}'" );
                        break;
                    default:
                        all_sents.Add( sent.ToArray() );
                        sent.Clear();
                        break;
                }
            }

            using var sw = new StreamWriter( inputFileName, append: false, Encoding.UTF8 );

            foreach ( var train_sent in all_sents )
            {
                foreach ( var (token, pos) in train_sent )
                {
                    sw.Write( token );
                    sw.Write( '\t' );
                    sw.WriteLine( pos );
                }
                sw.WriteLine();
            }
        }

        private static void ReplaceNumsOnPlaceholder( TokenizerConfig config, string inputFileName, string outputFileName, bool replaceNumsOnPlaceholder )
        {
            using var tokenizer = new Tokenizer( config, replaceNumsOnPlaceholder );

            Console.WriteLine( $"\r\n-------------------------------------------------\r\n file: '{inputFileName}' => '{outputFileName}'" );

            using var sr = new StreamReader( inputFileName );
            using var sw = new StreamWriter( outputFileName, append: false );

            var sent = new List< (string token, string pos) >();
            var ln = 0;
            var skiped_1 = 0;
            var skiped_2 = 0;
            var abc      = 0;
            for ( var line = sr.ReadLine(); line != null; line = sr.ReadLine() )
            {
                ln++;
                if ( !line.IsNullOrWhiteSpace() ) 
                { 
                    var idx = line.IndexOf( '\t' );
                    Debug.Assert( idx != -1 );
                    var token = line.Substring( 0, idx  );
                    var pos   = line.Substring( idx + 1 );

                    sent.Add( (token, pos) ); 
                    continue; 
                }

                if ( sent.Count == 1 )
                {
                    var token = sent[ 0 ].token;
                    if ( token.Length < 1 || token.All( c => char.IsPunctuation( c ) ) )
                    {
                        skiped_1++;
                        sent.Clear();
                        continue;
                    }
                }
                else if ( sent.Count == 2 )
                {
                    var token = sent[ 0 ].token;
                    if ( token.Length < 1 || token.All( c => char.IsPunctuation( c ) ) )
                    {
                        token = sent[ 1 ].token;
                        if ( token.Length < 1 || token.All( c => char.IsPunctuation( c ) ) )
                        {
                            skiped_2++;
                            sent.Clear();
                            continue;
                        }
                    }
                }

                for ( var i = 0; i < sent.Count; i++ )
                {
                    var t = sent[ i ];
                    if ( t.token.Length == 0 )
                    {
                        sent.RemoveAt( i ); i--;
                        continue;
                    }
                    var is_last_dot = xlat.IsDot( t.token[ ^1 ] );
                    if ( is_last_dot )
                    {
                        t.token += "XXX";
                    }
                    var token_words = tokenizer.Run_NoSentsAllocate( t.token );
                    if ( is_last_dot )
                    {
                        Debug.Assert( 1 < token_words.Count );
                        token_words.RemoveAt( token_words.Count - 1 );
                        t.token = t.token.Substring( 0, t.token.Length - "XXX".Length );
                    }
                    //---Debug.Assert( token_words.Count == 1 );
                    if ( token_words.Count == 0 )
                    {
                        sent.RemoveAt( i ); i--;
                    }
                    else if ( token_words.Count == 1 )
                    {
                        var w = token_words[ 0 ];
                        var v = w.valueOriginal;
                        if ( (w.extraWordType & ExtraWordType.Punctuation) == ExtraWordType.Punctuation )
                        {
                            if ( w.length == 1 )
                            {
                                var ch = w.valueOriginal[ 0 ];
                                switch ( ch )
                                {
                                    case ':': case '.': case ',': case ';': case '?': case '!': case '(': case ')': case '/': case '%': case '&': case '…': break;
                                    default:
                                        var ct = xlat.CHARTYPE_MAP[ ch ];
                                        if ( xlat.IsHyphen( ct ) )
                                        {
                                            if ( ch != '-' )
                                            {
                                                v = "-";
                                            }
                                        }
                                        else if ( (ct & CharType.IsQuote) == CharType.IsQuote )
                                        {
                                            switch ( ch )
                                            {
                                                case '\"': case '\'': case '[': case ']': break;
                                                default:
                                                    v = "\"";
                                                    break;
                                            }
                                        }
#if DEBUG && XXX
                                        else
                                        {
                                            switch ( ch )
                                            {
                                                case '[': case ']': case '{': case '}': break;
                                                default:
                                                    int ttttt = 0;
                                                    break;
                                            }
                                        } 
#endif
                                        break;
                                }
                            }
                            else if ( v == "''" )
                            {
                                v = "\"";
                            }
                            else if ( v == ",," )
                            {
                                v = ",";
                            }
                            else if ( v == "--" )
                            {
                                v = "-";
                            }
#if DEBUG && XXX
                            else if ( v != "..." && v != "//" && v != ".+" )
                            {
                                int tttttt = 0; 
                            }
#endif
                        }
                        sent[ i ] = (v, t.pos);
                    }
                    else
                    {
                        if ( t.pos == "Numeral" )
                        {
                            sent[ i ] = (token_words[ 0 ].valueOriginal, t.pos);
                            for ( var k = 1; k < token_words.Count; k++ )
                            {
                                sent.Insert( i + 1, (token_words[ k ].valueOriginal, t.pos) );
                            }
                            i += token_words.Count - 1;
                        }
                        else
                        {
                            if ( (token_words.Count == 2) && (token_words[ 0 ].posTaggerInputType == PosTaggerInputType.Num) )
                            {
                                sent[ i ] = (token_words[ 0 ].valueOriginal, "Numeral");
                                var pos = (token_words[ 1 ].valueOriginal == "мск") ? "Noun" : t.pos;
                                sent.Insert( i + 1, (token_words[ 1 ].valueOriginal, pos) );
                                i++;
                            }
                            else
                            {
                                //Debug.Assert( token_words.All( w => w.valueOriginal != Tokenizer.NUM_PLACEHOLDER ) );

                                //sent[ i ] = (t.token, t.pos);
                                abc++;

                                static (string t, string pos) get( word_t w, string pos )
                                {
                                    if ( w.posTaggerInputType == PosTaggerInputType.Num ) return (w.valueOriginal, "Numeral");
                                    if ( (w.extraWordType & ExtraWordType.Punctuation) == ExtraWordType.Punctuation ) return (w.valueOriginal, "Punctuation");
                                    return (w.valueOriginal, pos);
                                };
                                sent[ i ] = get( token_words[ 0 ], t.pos );
                                for ( var k = 1; k < token_words.Count; k++ )
                                {
                                    sent.Insert( i + 1, get( token_words[ k ], t.pos ) );
                                }
                                i += token_words.Count - 1;
                            }
                        }
                    }
                }

                var is_prev_num = false;
                for ( var i = 0; i < sent.Count; i++ )
                {
                    var (token, pos) = sent[ i ];
                    if ( token == Tokenizer.NUM_PLACEHOLDER )
                    {
                        if ( is_prev_num )
                        {
                            sent.RemoveAt( i );
                            i--;
                            continue;
                        }
                        is_prev_num = true;
                    }
                    else if ( is_prev_num )
                    {
                        is_prev_num = false;
                    }
                }
                foreach ( var (token, pos) in sent )
                {
                    sw.Write( token );
                    sw.Write( '\t' );
                    sw.WriteLine( pos );
                }
                sw.WriteLine();
                sent.Clear();
            }
            Console.WriteLine( $"{skiped_1}, {skiped_2}, {abc}" );
        }

        private static bool IsNullOrWhiteSpace( this string s ) => string.IsNullOrWhiteSpace( s );
    }
}
