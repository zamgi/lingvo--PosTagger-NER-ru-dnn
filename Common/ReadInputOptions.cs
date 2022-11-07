using System;
using System.IO;

using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    internal static class OptionsExtensions
    {
        public static (Options opts, string optsFileName) ReadInputOptions( string[] args, string DEFAULT_CONFIG_FILENAME = null ) => ReadInputOptions< Options >( args, null, DEFAULT_CONFIG_FILENAME );
        public static (T opts, string optsFileName) ReadInputOptions< T >( string[] args, string DEFAULT_CONFIG_FILENAME = null ) where T : Options, new() => ReadInputOptions< T >( args, null, DEFAULT_CONFIG_FILENAME );
        public static (T opts, string optsFileName) ReadInputOptions< T >( string[] args, Action< T > processOptsAction, string DEFAULT_CONFIG_FILENAME = null ) where T : Options, new()
        {
            const string PARENT_DIR = @"..\";

            #region [.read input params.]
            Logger.WriteLine( $"Command Line = '{string.Join( " ", args )}'" );

            var optsFileName = default(string);
            var opts = new T();
            ArgParser.Parse( args, opts, throwForUnknownParams: false );

            if ( (DEFAULT_CONFIG_FILENAME != null) && opts.ConfigFilePath.IsNullOrEmpty() )
            {
                     if ( File.Exists(              DEFAULT_CONFIG_FILENAME ) ) opts.ConfigFilePath =              DEFAULT_CONFIG_FILENAME;
                else if ( File.Exists( PARENT_DIR + DEFAULT_CONFIG_FILENAME ) ) opts.ConfigFilePath = PARENT_DIR + DEFAULT_CONFIG_FILENAME;
            }

            if ( !opts.ConfigFilePath.IsNullOrEmpty() )
            {
                Logger.WriteLine( $"Loading config file from '{opts.ConfigFilePath}'" );
                optsFileName = Path.GetFullPath( opts.ConfigFilePath );
                opts = JsonConvert.DeserializeObject< T >( File.ReadAllText( optsFileName ) );
            }
            if ( !opts.CurrentDirectory.IsNullOrEmpty() )
            {
                try { Environment.CurrentDirectory = opts.CurrentDirectory; }
                catch
                {
                    if ( opts.CurrentDirectory.StartsWith( PARENT_DIR ) )
                    {
                        Environment.CurrentDirectory = opts.CurrentDirectory.Substring( PARENT_DIR.Length );
                    }
                    else
                    {
                        throw;
                    }                        
                }
            }

            processOptsAction?.Invoke( opts );

            Logger.WriteLine( $"Configs: {JsonConvert.SerializeObject( opts, Formatting.Indented, CreateJsonSerializerSettings() )}" );

            return (opts, optsFileName);
            #endregion
        }

        private static JsonSerializerSettings CreateJsonSerializerSettings() => new JsonSerializerSettings()
        {
            NullValueHandling = NullValueHandling.Ignore,
            Converters        = new[] { new StringEnumConverter() },
        };
    }
}

