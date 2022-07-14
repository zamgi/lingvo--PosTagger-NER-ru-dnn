using System;
using System.IO;
using System.Text;

namespace Lingvo.PosTagger.Utils
{
    /// <summary>
    /// 
    /// </summary>
    public static class Logger
    {
        /// <summary>
        /// 
        /// </summary>
        public enum Level 
        { 
            Error,
            Warn, 
            Info 
        };

        private static string _LogFileName;
        private static StreamWriter _SW;

        #region comm.
        //public static void WriteLine( string s, params object[] args ) => WriteLine( Level.Info, s, args );
        //public static void WriteErrorLine( string s, params object[] args ) => WriteLine( Level.Error, ConsoleColor.Red, s, args );
        //public static void WriteWarnLine( string s, params object[] args ) => WriteLine( Level.Warn, ConsoleColor.Yellow, s, args );
        //public static void WriteInfoLine( string s, params object[] args ) => WriteLine( Level.Info, s, args );
        //public static void WriteInfoLine( ConsoleColor color, string s, params object[] args ) => WriteLine( Level.Info, color, s, args );
        //public static void WriteLine( Level level, string s, params object[] args )
        //{
        //    var sb = new StringBuilder( 0x100 ).Append( level ).Append(',').Append( DateTime.Now );
        //    if ( args.Length == 0 )
        //        sb.Append( s );
        //    else
        //        sb.AppendFormat( s, args );
        //    var msg = sb.ToString();

        //    if ( level != Level.Info )
        //        Console.Error.WriteLine( msg );
        //    else
        //        Console.WriteLine( msg );

        //    _SW?.WriteLine( msg );
        //}
        //public static void WriteLine( Level level, ConsoleColor color, string s, params object[] args )
        //{
        //    var sb = new StringBuilder( 0x100 ).Append( level ).Append( ',' ).Append( DateTime.Now );
        //    if ( args.Length == 0 )
        //        sb.Append( s );
        //    else
        //        sb.AppendFormat( s, args );
        //    var msg = sb.ToString();

        //    var fc = Console.ForegroundColor;
        //    Console.ForegroundColor = color;
        //    if ( level != Level.Info )
        //        Console.Error.WriteLine( msg );
        //    else
        //        Console.WriteLine( msg );
        //    Console.ForegroundColor = fc; //---Console.ResetColor();

        //    _SW?.WriteLine( msg );
        //}
        #endregion

        public static void WriteLine( string s ) => WriteLine( Level.Info, s );        
        public static void WriteErrorLine( string s ) => WriteLine( Level.Error, ConsoleColor.Red, s );        
        public static void WriteWarnLine( string s ) => WriteLine( Level.Warn, ConsoleColor.Yellow, s );        
        public static void WriteInfoLine( string s ) => WriteLine( Level.Info, s );        
        public static void WriteInfoLine( ConsoleColor color, string s ) => WriteLine( Level.Info, color, s  );

        public static void WriteLine( Level level, string s )
        {
            var sb  = new StringBuilder( 0x100 ).Append( level.AsText() ).Append( ',' ).Append( DateTime.Now ).Append( ' ' ).Append( s );
            var msg = sb.ToString();

            if ( level != Level.Info )
                Console.Error.WriteLine( msg );
            else
                Console.WriteLine( msg );

            _SW?.WriteLine( msg );
        }
        public static void WriteLine( Level level, ConsoleColor color, string s )
        {
            var sb  = new StringBuilder( 0x100 ).Append( level.AsText() ).Append( ',' ).Append( DateTime.Now ).Append( ' ' ).Append( s );
            var msg = sb.ToString();

            var fc = Console.ForegroundColor;
            Console.ForegroundColor = color;
            if ( level != Level.Info )
                Console.Error.WriteLine( msg );
            else
                Console.WriteLine( msg );
            Console.ForegroundColor = fc; //---Console.ResetColor();

            _SW?.WriteLine( msg );
        }

        private static string AsText( this Level level )
        {
            switch ( level )
            {
                case Level.Error: return ("Error");
                case Level.Warn : return ("Warn");
                case Level.Info : return ("Info");
                default: return (level.ToString());
            }
        }

        public static void Close()
        {
            if ( _SW != null )
            {
                _SW.Close();
                _SW = null;
            }
        }

        public static string LogFile
        {
            get => _LogFileName;
            set
            {
                if ( _LogFileName == value )
                    return;

                if ( _SW != null )
                {
                    _SW.Close();
                    _SW = null;
                }

                _LogFileName = value;
                if ( _LogFileName != null )
                {
                    _SW = new StreamWriter( _LogFileName, append: true, Encoding.UTF8 ) { AutoFlush = true };
                }
            }
        }
    }
}
