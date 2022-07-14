using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

using Newtonsoft.Json;
#if DEBUG
using Newtonsoft.Json.Converters;
#endif
using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Utils;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;
using _OptionsAllowedChangingDelege_ = Lingvo.PosTagger.Applications.OptionsAllowedChanging.ChangingDelege;

namespace Lingvo.PosTagger.ModelBuilder
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class OptionsFileChangingWatcher : IDisposable
    {
        #region [.ctor().]
        public event _OptionsAllowedChangingDelege_ OnOptionsAllowedChanging;

        private FileSystemWatcher      _Fsw;
        private OptionsAllowedChanging _LastOpts;
        private string                 _OptionsFullFileName;
        private string                 _OptionsFileName;        
        public OptionsFileChangingWatcher( Options initOpts, string optionsFileName, _OptionsAllowedChangingDelege_ optionsAllowedChangingAction = null )
        {            
            if ( optionsFileName.IsNullOrWhiteSpace() || !File.Exists( optionsFileName ) ) throw (new FileNotFoundException( $"Options file doesn't exists: {(optionsFileName.IsNullOrWhiteSpace() ? "NULL" : $"'{optionsFileName}'")}.", optionsFileName ));
            if ( initOpts == null ) initOpts = new Options();
            //-----------------------------------------------//

            _OptionsFullFileName = Path.GetFullPath( optionsFileName );
            _OptionsFileName     = Path.GetFileName( _OptionsFullFileName );
            var dir = Path.GetDirectoryName( _OptionsFullFileName );

            if ( optionsAllowedChangingAction != null ) OnOptionsAllowedChanging += optionsAllowedChangingAction;
            _LastOpts = initOpts.ToOptionsAllowedChanging();

            Console.WriteLine( $"Start watching file: '{_OptionsFullFileName}'..." );
            Console.WriteLine( $"(Watching props: '{string.Join( "', '", GetWatchingProps() )}')" );

            _Fsw = new FileSystemWatcher( dir, _OptionsFileName )
            {
                EnableRaisingEvents = true,
                NotifyFilter        = NotifyFilters.Size | NotifyFilters.LastWrite | NotifyFilters.Security
            };

            _Fsw.Error   += _Fsw_Error;
            _Fsw.Changed += _Fsw_Changed;            
        }
        public static bool TryCreate( Options initOpts, string optionsFileName, out OptionsFileChangingWatcher ofcw, params _OptionsAllowedChangingDelege_[] optionsAllowedChangingActions )
        {
            try
            {
                ofcw = new OptionsFileChangingWatcher( initOpts, optionsFileName );
                if ( optionsAllowedChangingActions.AnyEx() )
                {
                    foreach ( var a in optionsAllowedChangingActions )
                    {
                        if ( a != null ) ofcw.OnOptionsAllowedChanging += a;
                    }
                }
                return (true);
            }
            catch ( Exception ex )
            {
                Debug.WriteLine( ex );
            }
            ofcw = default;
            return (false);
        }
        public static OptionsFileChangingWatcher TryCreate( Options initOpts, string optionsFileName, params _OptionsAllowedChangingDelege_[] optionsAllowedChangingActions ) => TryCreate( initOpts, optionsFileName, out var ofcw, optionsAllowedChangingActions ) ? ofcw : null;

        public void Dispose()
        {
            _Fsw.EnableRaisingEvents = false;
            _Fsw.Dispose();
        }
        #endregion

        private static IEnumerable< string > GetWatchingProps() => typeof(OptionsAllowedChanging).GetProperties().Select( p => p.Name );
        private void _Fsw_Error( object sender, ErrorEventArgs e ) => Console.WriteLine( $"[FileSystemWatcher::Error => {e.GetException()}]" );
        private void _Fsw_Changed( object sender, FileSystemEventArgs e )
        {
            if ( e.ChangeType.HasChanged() && _OptionsFileName.IsEqualIgnoreCase( e.Name ) && _OptionsFullFileName.IsEqualIgnoreCase( e.FullPath ) )
            {
                try
                {
                    var raw_opts = JsonConvert.DeserializeObject< Options >( File.ReadAllText( e.FullPath ) );
                    var opts = raw_opts.ToOptionsAllowedChanging();
                    if ( !opts.IsEqual( _LastOpts ) )
                    {
                        _LastOpts = opts;
                        try
                        {
#if DEBUG
                            var jss = new JsonSerializerSettings()
                            {
                                NullValueHandling = NullValueHandling.Ignore,
                                Converters        = new[] { new StringEnumConverter() },
                            };
                            Logger.WriteLine( $"Changed options-file: {JsonConvert.SerializeObject( raw_opts, Formatting.Indented, jss )}" );
#else
                            Console.WriteLine( "Changed options-file." );
#endif
                            OnOptionsAllowedChanging?.Invoke( opts );
                        }
                        catch ( Exception ex )
                        {
                            Console.WriteLine( $"[FileSystemWatcher::Changed => {ex}]" );
                        }                        
                    }
                    else
                    {
                        Console.WriteLine( "Idle changed options-file." );
                    }
                }
                catch ( Exception ex )
                {
                    Console.WriteLine( $"[FileSystemWatcher::Changed => {ex}]" );
                }
            }
        }
    }

    /// <summary>
    /// 
    /// </summary>
    internal static class OptionsFileChangingWatcher_Extensions
    {
        [M(O.AggressiveInlining)] public static bool IsChanged( this WatcherChangeTypes wct ) => (wct == WatcherChangeTypes.Changed);
        [M(O.AggressiveInlining)] public static bool HasChanged( this WatcherChangeTypes wct ) => (wct & WatcherChangeTypes.Changed) == WatcherChangeTypes.Changed;
        [M(O.AggressiveInlining)] public static bool IsEqualIgnoreCase( this string s1, string s2 ) => (string.Compare( s1, s2, true ) == 0);

        [M(O.AggressiveInlining)] 
        public static bool IsEqual( this in OptionsAllowedChanging x, in OptionsAllowedChanging y ) => (x.BatchSize             == y.BatchSize) &&
                                                                                                       (x.MaxEpochNum           == y.MaxEpochNum) &&
                                                                                                       (x.Valid_RunEveryUpdates == y.Valid_RunEveryUpdates);
        [M(O.AggressiveInlining)] public static OptionsAllowedChanging ToOptionsAllowedChanging( this Options opts ) => new OptionsAllowedChanging()
        {
            BatchSize             = opts.BatchSize,
            MaxEpochNum           = opts.MaxEpochNum,
            Valid_RunEveryUpdates = opts.Valid_RunEveryUpdates,
        };
    }
}
