using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler
{
    /// <summary>
    /// 
    /// </summary>

    public sealed class KernelDiskCache
    {
        private readonly string _CacheDir;
        private readonly Dictionary<string, byte[]> _MemoryCachedKernels;

        public KernelDiskCache( string cacheDir )
        {
            _CacheDir = cacheDir;
            _MemoryCachedKernels = new Dictionary<string, byte[]>();
            if ( !Directory.Exists( cacheDir ) )
            {
                Directory.CreateDirectory( cacheDir );
            }
        }

        /// <summary>
        /// Deletes all kernels from disk if they are not currently loaded into memory. Calling this after
        /// calling TSCudaContext.Precompile() will delete any cached .ptx files that are no longer needed
        /// </summary>
        public void CleanUnused()
        {
            foreach ( var file in Directory.EnumerateFiles( _CacheDir ) )
            {
                var key = KeyFromFilePath( file );
                if ( !_MemoryCachedKernels.ContainsKey( key ) )
                {
                    try { File.Delete( file ); } catch {; }
                }
            }
        }

        public byte[] Get( string fullSourceCode, Func<string, byte[]> compile )
        {
            var key = KeyFromSource( fullSourceCode );
            if ( _MemoryCachedKernels.TryGetValue( key, out var ptx ) )
            {
                return (ptx);
            }
            else if ( TryGetFromFile( key, out ptx ) )
            {
                _MemoryCachedKernels.Add( key, ptx );
                return (ptx);
            }
            else
            {
                WriteCudaCppToFile( key, fullSourceCode );

                ptx = compile( fullSourceCode );
                _MemoryCachedKernels.Add( key, ptx );
                WriteToFile( key, ptx );

                return (ptx);
            }
        }

        private void WriteToFile( string key, byte[] ptx )
        {
            var filePath = FilePathFromKey( key );

            Logger.WriteLine( $"Writing PTX code to '{filePath}'." );
            File.WriteAllBytes( filePath, ptx );
        }

        private void WriteCudaCppToFile( string key, string sourceCode )
        {
            var filePath = FilePathFromKey( key ) + ".cu";

            Logger.WriteLine( $"Writing cuda source code to '{filePath}'." );
            File.WriteAllText( filePath, sourceCode );
        }

        private bool TryGetFromFile( string key, out byte[] ptx )
        {
            var filePath = FilePathFromKey( key );
            if ( !File.Exists( filePath ) )
            {
                ptx = null;
                return (false);
            }

            ptx = File.ReadAllBytes( filePath );
            return (true);
        }

        private string FilePathFromKey( string key ) => Path.Combine( _CacheDir, key + ".ptx" );

        private static string KeyFromFilePath( string filepath )
        {
            const string PTX = ".ptx";
            if ( filepath.EndsWith( PTX, StringComparison.InvariantCultureIgnoreCase ) )
            {
                filepath = filepath.Substring( 0, filepath.Length - PTX.Length );
            }

            const string CU = ".cu";
            if ( filepath.EndsWith( CU, StringComparison.InvariantCultureIgnoreCase ) )
            {
                filepath = filepath.Substring( 0, filepath.Length - CU.Length );
            }

            return Path.GetFileNameWithoutExtension( filepath );
        }

        private static string KeyFromSource( string fullSource )
        {
            var fullKey = fullSource.Length + fullSource;
            using ( var sha1 = SHA1.Create() )
            {
                return (BitConverter.ToString( sha1.ComputeHash( Encoding.UTF8.GetBytes( fullKey ) ) ).Replace( "-", string.Empty ));
            }
        }
    }
}
