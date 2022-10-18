using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Tokenizing;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.ConsoleDemo
{
    /// <summary>
    /// 
    /// </summary>
    internal static class Program
    {
        /// <summary>
        /// 
        /// </summary>
        private enum ValidateModeEnum
        {
            _default,
            one_model__by_many_valid_files,
            all_moldels__by_one_valid_file,
        }
        /// <summary>
        /// 
        /// </summary>
        private sealed class Config : Options
        {
            [Arg(nameof(Validate_Mode)     )] public ValidateModeEnum Validate_Mode;
            [Arg(nameof(ValidFilesRootPath))] public string           ValidFilesRootPath; // if ["Validate_Mode" == "one_model__by_many_valid_files"]
            [Arg(nameof(ModelsRootPath)    )] public string           ModelsRootPath;     // if ["Validate_Mode" == "all_moldels__by_one_valid_file"]
        }

        private static void Main( string[] args )
        {
            try
            {
                var opts = OptionsExtensions.ReadInputOptions< Config >( args, "predict.json" ).opts;

                //test_predict_1( opts );

                if ( args.Any( a => a == "valid.json" ) )
                {
                    switch ( opts.Validate_Mode )
                    {
                        case ValidateModeEnum.one_model__by_many_valid_files:
                            validate_one_model__by_many_valid_files( opts, opts.ValidFilesRootPath ); // @"..\..\..\..\[package's]\Lingvo.PosTagger_and_NER\RU\valid\" );
                            break;

                        case ValidateModeEnum.all_moldels__by_one_valid_file:
                            validate_all_moldels__by_one_valid_file( opts, opts.ModelsRootPath ); // @"..\[resources]\models\ner\" );
                            break;

                        default: 
                            validate( opts ); 
                            break;
                    }
                }
                else
                {
                    Run_Predict( opts );
                }
            }
            catch ( Exception ex )
            {
                Logger.WriteErrorLine( Environment.NewLine + ex + Environment.NewLine );
            }
        }

        private static void validate( Options opts )
        {
            var vr = Validator.Run_Validate( opts );

            var fc = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine();
            Console.WriteLine( "---------------------------------------------------------" );
            Console.WriteLine( $"m: '{Path.GetFileName( opts.ModelFilePath )}',  (v: '{Path.GetFileName( opts.ValidCorpusPath )}')" );
            Console.WriteLine( "---------------------------------------------------------" );
            Console.WriteLine( vr.ToString().TrimEnd( '\r', '\n' ) );
            Console.WriteLine( "---------------------------------------------------------" );
            Console.WriteLine();
            Console.ForegroundColor = fc;

            GC_Collect();
        }
        private static void validate_one_model__by_many_valid_files( Options opts, string validFilesRootPath )
        {
            using var cts = Console_CancelKeyPress_Breaker.Create( cancelMessage: null );

            using var sw = new StreamWriter( @$"E:\scores-[{Path.GetFileNameWithoutExtension( opts.ModelFilePath )}].txt", append: true );

            //var validFiles = Directory.EnumerateFiles( validFilesRootPath, "valid_*.txt", SearchOption.TopDirectoryOnly );
            var validFiles = new[]
            {
                "valid_ner_ru__(0.5M).txt",
                "valid_ner_ru__(1.0M).txt",
                "valid_ner_ru__(2.0M).txt",
                "valid_ner_ru__(3.0M).txt",
                "valid_ner_ru__(4.0M).txt",
                "valid_ner_ru__(5.0M).txt",
                "valid_ner_ru__(6.0M).txt",
                "valid_ner_ru__(7.0M).txt",
                "valid_ner_ru__(7.5M).txt",
            };
            foreach ( var ValidCorpusPath in validFiles )
            {
                if ( cts.IsCancellationRequested ) break;

                opts.ValidCorpusPath = Path.Combine( validFilesRootPath, ValidCorpusPath );
                //opts.ValidCorpusPath = ValidCorpusPath;
                var vr = Validator.Run_Validate( opts );

                sw.WriteLine( "---------------------------------------------------------" );
                sw.WriteLine( $"m: '{Path.GetFileName( opts.ModelFilePath )}',  (v: '{Path.GetFileName( ValidCorpusPath )}')" );
                sw.WriteLine( "---------------------------------------------------------" );
                sw.WriteLine( vr.ToString().TrimEnd('\r', '\n') );
                sw.WriteLine( "---------------------------------------------------------" );
                sw.WriteLine();
                sw.Flush();

                GC_Collect();
            }
        }
        private static void validate_all_moldels__by_one_valid_file( Options opts, string modelsRootPath )
        {
            using var cts = Console_CancelKeyPress_Breaker.Create( cancelMessage: null );

            using var sw = new StreamWriter( Path.Combine( modelsRootPath, $"scores__[{Path.GetFileNameWithoutExtension( opts.ValidCorpusPath )}].txt" ), append: false );

            foreach ( var ModelFilePath in Directory.EnumerateFiles( modelsRootPath, "*.s2s", SearchOption.TopDirectoryOnly ) )
            {
                if ( cts.IsCancellationRequested ) break;

                opts.ModelFilePath = ModelFilePath;
                var vr = Validator.Run_Validate( opts );

                sw.WriteLine( "---------------------------------------------------------" );
                sw.WriteLine( $"m: '{Path.GetFileName( ModelFilePath )}',  (v: '{Path.GetFileName( opts.ValidCorpusPath )}')" );
                sw.WriteLine( "---------------------------------------------------------" );
                sw.WriteLine( vr.ToString().TrimEnd('\r', '\n') );
                sw.WriteLine( "---------------------------------------------------------" );
                sw.WriteLine();
                sw.Flush();

                GC_Collect();
            }
        }

        private static void test_predict_1( Options opts )
        {
            var tokenizerConfig = new TokenizerConfig( opts.SentSplitterResourcesXmlFilename, opts.UrlDetectorResourcesXmlFilename );
            using var tokenizer = new Tokenizer( tokenizerConfig, replaceNumsOnPlaceholder: true );

            void tok( string txt )
            {
                var www   = tokenizer.Run_NoSentsAllocate( txt );
                var sss_1 = tokenizer.Run_SimpleSentsAllocate_UnbendList( txt );
                //var sss_2 = tokenizer.__Run___v0( txt );
            };

            tok( "пятьдесят оттенков серого э.л. джеймс." );
            tok( "По данным следователей, в июле 2010г. военный чиновник " );
            tok( "По данным следователей, в июле 2010 г. военный чиновник " );
            tok( "По данным следователей, в июле 2010 г . военный чиновник " );


            var predictor = new Predictor( opts );
            
            IList< word_t > predict_routine( string text )
            {
                var words = tokenizer.Run_SimpleSentsAllocate_UnbendList( text );

                var maxEndingLength = ((0 < opts.MaxEndingLength) ? opts.MaxEndingLength : int.MaxValue);
                var input_words = words.Select( w => Tokenizer.ToPosTaggerToken( w, maxEndingLength ) ).ToList();
                var (output_words, clsInfo) = predictor.Predict( input_words );

                Debug.WriteLine( $"WordsInDictRatio: {clsInfo.WordsInDictRatio}" );
                foreach ( var wc in clsInfo.WordClasses )
                {
                    Debug.WriteLine( $"'{wc.Word}' => {string.Join( ", ", wc.Classes.Select( c => $"{c.ClassName} ({c.Probability:F4})" ) )}" );
                }

                words.SetPosTaggerOutputType( output_words );

                words.Print2Console( text );

                return (words);
            };

            Console.WriteLine();
            var words = predict_routine( "По данным следователей, в июле 2010г. военный чиновник " );
                        predict_routine( "По данным следователей, в июле 2010 г. военный чиновник " );
                        predict_routine( "По данным следователей, в июле 2010 г . военный чиновник " );
                        predict_routine( "По данным следователей, в июле 2010г. военный чиновник отдал подчиненному \"заведомо преступный приказ\" о заключении лицензионных договоров с компаниями \"Чарт-Пилот\" и \"Транзас\". Им необоснованно были переданы права на использование в коммерческих целях навигационных морских карт, являвшихся интеллектуальной собственностью РФ. В результате ущерб составил более 9,5 млн руб.");

                //words = predict_routine( @"Глокая куздра штеко будланула бокра и курдячит бокрёнка." );
                //words = predict_routine( @"Гло́кая ку́здра ште́ко будлану́ла бо́кра и курдя́чит бокрёнка." );
                //words = predict_routine( @"Варкалось . Хливкие шорьки пырялись по наве, и хрюкотали зелюки, как мюмзики в мове." );
                words = predict_routine( "123-3453-3456-3456 коровы." );
                words = predict_routine( "123 4567 890 коровы." );
                words = predict_routine( "zxczxcv https://localhost:7701/ xzxzxzzxzx." );
            
            
                words = predict_routine( @"Маша руками мыла посуду." );
                words = predict_routine( @"Маша руками Вася звал на помощь." );
                words = predict_routine( @"Реки стали красными." );
                words = predict_routine( @"Реки стали красными потоками текли." );

                words = predict_routine( @"Вася, маша руками и коля дрова, морочил голову." );
                words = predict_routine( @"Вася, Маша и Коля пошли гулять." );                                
        }

        private static void Print2Console( this IList< word_t > words, string text )
        {
            var max_len = words.Max( w => w.length ) + 3;
            string get_original_text_word_4_print( word_t w )
            {
                var t = text?.Substring( w.startIndex, w.length ) ?? w.valueOriginal;
                return ($"'{t}'{new string( ' ', max_len - t.Length )} {w.GetOutputType()}");
            };
            Console.WriteLine( string.Join( "\r\n", words.Select( w => get_original_text_word_4_print( w ) ) ) + "\r\n---------------------------------------------\r\n" );
        }

        private static void Run_Predict( Options opts )
        {
            if ( opts.OutputTestFile.IsNullOrEmpty() ) opts.OutputTestFile = "output_pos_tagger_ru.txt";

            Logger.WriteLine( $"Test model '{opts.ModelFilePath}' by input corpus '{opts.InputTestFile}'" );
            Logger.WriteLine( $"Output to '{Path.GetFullPath( opts.OutputTestFile )}'" );
            Console.WriteLine();

            var predictor = new Predictor( opts );

            var data_sents_raw = File.ReadLines( opts.InputTestFile ).Where( line => !line.IsNullOrWhiteSpace() );

            using var sw = new StreamWriter( opts.OutputTestFile, append: false, Encoding.UTF8 );
            var n = 0;
#if DEBUG
            var po = new ParallelOptions() { MaxDegreeOfParallelism = 1 };
#else
            var po = new ParallelOptions() { MaxDegreeOfParallelism = 1 }; // Environment.ProcessorCount };
#endif
            var maxEndingLength = ((0 < opts.MaxEndingLength) ? opts.MaxEndingLength : int.MaxValue);
            var tokenizerConfig = new TokenizerConfig( opts.SentSplitterResourcesXmlFilename, opts.UrlDetectorResourcesXmlFilename );
            Parallel.ForEach( data_sents_raw, po,
            () => new Tokenizer( tokenizerConfig, replaceNumsOnPlaceholder: true ),
            (line, _, _, tokenizer) =>
            {
                if ( line.IsNullOrWhiteSpace() ) return (tokenizer);

                var words = tokenizer.Run_NoSentsAllocate( line );
                if ( words.Count <= 0 ) return (tokenizer);
                
                var input_words = words.Select( w => Tokenizer.ToPosTaggerToken( w, maxEndingLength ) ).ToList(); //---var input_words  = words.Select( w => w.valueOriginal ).ToList();
                var (output_words, clsInfo) = predictor.Predict( input_words );

                Debug.WriteLine( $"WordsInDictRatio: {clsInfo.WordsInDictRatio}" );
                foreach ( var wc in clsInfo.WordClasses )
                {
                    Debug.WriteLine( $"'{wc.Word}' => {string.Join(", ", wc.Classes.Select( c => $"{c.ClassName} ({c.Probability:F4})" ) )}" );
                }

                words.SetPosTaggerOutputType( output_words );
                //NerPostMerger.Run_Merge( words );

                lock ( sw )
                {
                    sw.WriteLine( $"{++n})." ); Console.WriteLine( $"{n})." );

                    if ( output_words.Count == input_words.Count )
                    {
                        string get_word_value( int i ) => /*input_words[ i ]*/ Tokenizer.GetOriginalValue( words[ i ], line );

                        Span< int > max_lens = stackalloc int[ output_words.Count ];

                        var len = output_words.Count - 1;
                        for ( var i = 0; i <= len; i++ )
                        {
                            max_lens[ i ] = Math.Max( get_word_value( i ).Length, output_words[ i ].Length ) + 1;
                        }

                        for ( var i = 0; i <= len; i++ )
                        {
                            var s = get_word_value( i ).PadRight( max_lens[ i ] );
                            sw.Write( s ); Console.Write( s );
                        }
                        sw.WriteLine(); Console.WriteLine();

                        for ( var i = 0; i <= len; i++ )
                        {
                            var s = output_words[ i ];
                            if ( s.Length == 1 && s[ 0 ] == 'O' ) s = "-";
                            s = s.PadRight( max_lens[ i ] );
                            sw.Write( s ); Console.Write( s );
                        }
                        sw.WriteLine(); Console.WriteLine();
                    }
                    else
                    {
                        var word_values = /*input_words*/words.Select( w => w.valueOriginal );
                        sw.WriteLine( string.Join( " ", word_values ) ); Console.WriteLine( string.Join( " ", word_values ) );
                        sw.WriteLine( string.Join( " ", output_words ) ); Console.WriteLine( string.Join( " ", output_words ) );
                    }
                    sw.WriteLine(); Console.WriteLine();
                    sw.Flush();
                }
                return (tokenizer);
            },
            (tokenizer) => tokenizer.Dispose()
            );
        }

        private static void GC_Collect()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
    }
}
