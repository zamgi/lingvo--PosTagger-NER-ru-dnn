using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

using Lingvo.PosTagger.WebService;

namespace Lingvo.PosTagger.WebServiceTestApp
{
    /// <summary>
    /// 
    /// </summary>
    internal static class Program
    {
        private static string BASE_URL => ConfigurationManager.AppSettings[ "BASE_URL" ];

        private static async Task Main( string[] args )
        {
            try
            {
                Console.WriteLine( $"BASE_URL: '{BASE_URL}'\r\n" );

                using var httpClient = new HttpClient();
                var client = new PosTaggerWebServiceClient( httpClient, BASE_URL );

                await Run__Test( client, MAX_FILE_SIZE_IN_MB: 25 ).CAX();
            }
            catch ( Exception ex )
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine( "ERROR: " + ex );
                Console.ResetColor();
            }
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine( "\r\n\r\n[.....finita.....]\r\n\r\n" );
            Console.ResetColor();
            Console.ReadLine();
        }

        private static IEnumerable< string > GetFile_4_Processing( CancellationTokenSource cts, bool print2ConsoleTitle, int MAX_FILE_SIZE_IN_BYTES )
        {
            //var paths = DriveInfo.GetDrives().Where( di => di.DriveType == DriveType.Fixed ).Select( di => di.RootDirectory.FullName ).ToArray();
            //
            var paths = new[] { @"..\..\..\[resources]\input-text\" };
            //---------------------------------------------------------------//

            var file_extensions = new HashSet< string >( new[] { ".txt" }, StringComparer.InvariantCultureIgnoreCase ); 

            if ( print2ConsoleTitle )
            {
                Console.Title = $"start search files ('{string.Join( "', '", file_extensions )}') by paths: '{string.Join( "', '", paths )}'...";
            }

            var eo = new EnumerationOptions()
            {
                IgnoreInaccessible       = true,
                RecurseSubdirectories    = false, //true,
                ReturnSpecialDirectories = false, //true,
            };
            var seq = Enumerable.Empty< string >();
            foreach ( var path in paths )
            {
                seq = seq.Concat( Directory.EnumerateFiles( path, "*.*", eo ) );
            }

            var n  = 0;
            var sw = Stopwatch.StartNew();
            foreach ( var fileName in seq )
            {
                if ( cts.IsCancellationRequested ) yield break;

                if ( print2ConsoleTitle && (((++n % 5_000) == 0) || (1_500 <= sw.ElapsedMilliseconds)) )
                {
                    Console.Title = fileName;
                    sw.Restart();
                }
                var fi = new FileInfo( fileName );
                if ( !file_extensions.Contains( fi.Extension ?? string.Empty ) )
                {
                    continue;
                }
                if ( MAX_FILE_SIZE_IN_BYTES < fi.Length )
                {
                    continue;
                }
                yield return (fileName); 
            }
        }
        private static async Task Run__Test( PosTaggerWebServiceClient client, int MAX_FILE_SIZE_IN_MB )
        {
            var fileNumber = 0;

            var sw = Stopwatch.StartNew();
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => cts.Cancel_NoThrow( e );

            var machineName = Environment.MachineName;
            var files = GetFile_4_Processing( cts, print2ConsoleTitle: true, (1024 * 1024 * MAX_FILE_SIZE_IN_MB) ).ToList();
            var filesCount = files.Count;
#if DEBUG
            var dop = 1; //Environment.ProcessorCount;
#else
            var dop = Environment.ProcessorCount; //1; //
#endif            
            await files.ForEachAsync_( dop, cts.Token, async (fileName, ct) =>
            {
                if ( !TryReadAllText( fileName, out var text ) )
                {
                    return;
                }

                var result = await client.Run( text, ct ).CAX();

                Console.Title = $"processed files: {Interlocked.Increment( ref fileNumber )} of {filesCount}...";
                var cnt = result.Sents?.Sum( snt => snt.Tuples.Count );
                Console.WriteLine( $"processed '{fileName}' => PosTagger: {cnt.GetValueOrDefault()}" );

                //---Thread.Sleep( 100 );
            })
            .CAX();

            Console.Title = $"total processed files: {fileNumber}, (elapsed: {sw.Elapsed()}).";
        }
        private static bool TryReadAllText( string fileName, out string text )
        {
            try
            {
                text = File.ReadAllText( fileName );
                return (!text.IsNullOrWhiteSpace());
            }
            catch ( Exception ex )
            {
                Debug.WriteLine( ex );
            }
            text = default;
            return (false);
        }
    }

    /// <summary>
    /// 
    /// </summary>
    internal static class Extensions
    {
        public static bool IsNullOrEmpty( this string s ) => string.IsNullOrEmpty( s );
        public static bool IsNullOrWhiteSpace( this string s ) => string.IsNullOrWhiteSpace( s );

        public static TimeSpan Elapsed( this Stopwatch sw )
        {
            sw.Stop();
            return (sw.Elapsed);
        }
        public static void Cancel_NoThrow( this CancellationTokenSource cts, ConsoleCancelEventArgs e = null )
        {
            try
            {
                if ( e != null ) e.Cancel = true;
                cts.Cancel();
            }
            catch ( Exception ex )
            {
                Debug.WriteLine( ex );//suppress
            }
        }
        public static T[] ToArray< T >( this IEnumerable< T > seq, int len )
        {
            var a = new T[ len ];
            var i = 0;
            foreach ( var t in seq )
            {
                a[ i++ ] = t;
            }
            return (a);
        }
        public static bool StartsWith_Ex( this string s1, string s2 ) => s1.StartsWith( s2, StringComparison.InvariantCultureIgnoreCase );
        public static bool EndsWith_Ex( this string s1, string s2 ) => s1.EndsWith( s2, StringComparison.InvariantCultureIgnoreCase );
    }
}
