using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;

namespace Lingvo.PosTagger.Utils
{
    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
    public sealed class Arg : Attribute
    {
        public Arg( string name, bool optional = true ) : this( name, name, optional ) { }
        public Arg( string name, string title, bool optional = true )
        {
            Name     = name ?? throw (new ArgumentNullException());
            Title    = string.IsNullOrEmpty( title ) ? name : title;
            Optional = optional;
        }

        public string Title    { get; }
        public bool   Optional { get; }
        public string Name     { get; }

        public string UsageLineText() => (Optional ? "[" : null) + ($"-{Name}: {Title}") + (Optional ? "]" : null);

        public override string ToString() => $"'{Name}', '{Title}', optional: {Optional.ToString().ToUpper()}";
    }

    /// <summary>
    /// 
    /// </summary>
    public sealed class ArgField
    {
        private object    _Obj;
        private FieldInfo _FieldInfo;
        private Arg       _Arg;
        private bool      _IsSet;

        public ArgField( object obj, FieldInfo fieldInfo, Arg arg )
        {
            _Obj       = obj;
            _FieldInfo = fieldInfo;
            _Arg       = arg;
        }

        public Arg Arg => _Arg;

        public void Set( string val )
        {
            try
            {
                if ( _FieldInfo.FieldType == typeof(string) )
                {
                    _FieldInfo.SetValue( _Obj, val );
                }
                else
                {
                    Type argumentType = (_FieldInfo.FieldType.IsGenericType && (_FieldInfo.FieldType.GetGenericTypeDefinition() == typeof(Nullable<>))) ? _FieldInfo.FieldType.GenericTypeArguments[ 0 ]
                                                                                                                                                        : _FieldInfo.FieldType;
                    MethodInfo mi = argumentType.GetMethod( "Parse", new Type[] { typeof(string) } );
                    if ( mi != null )
                    {
                        var v = mi.Invoke( null, new object[] { val } );
                        _FieldInfo.SetValue( _Obj, v );
                    }
                    else if ( argumentType.IsEnum )
                    {
                        var v = Enum.Parse( _FieldInfo.FieldType, val );
                        _FieldInfo.SetValue( _Obj, v );
                    }
                }
                _IsSet = true;
            }
            catch ( Exception ex )
            {
                throw (new ArgumentException( $"Failed to set value of '{_Arg}', Error: '{ex.Message}', Call Stack: '{ex.StackTrace}'." ));
            }
        }
        public void Validate()
        {
            if ( !_Arg.Optional && !_IsSet )
            {
                throw (new ArgumentException( $"Failed to specify value for required '{_Arg}'." ));
            }
        }

        public override string ToString() => _Arg.ToString();
    }

    /// <summary>
    /// 
    /// </summary>
    public static class ArgParser
    {
        public static IReadOnlyCollection< ArgField > Parse( string[] args, object obj, bool throwForUnknownParams )
        {
            var args_dict = new Dictionary< string, ArgField >( 0x100, StringComparer.InvariantCultureIgnoreCase );
            foreach ( FieldInfo fi in obj.GetType().GetFields( BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance ) )
            {
                foreach ( Arg arg in fi.GetCustomAttributes( typeof(Arg), true )/*.Cast< Arg >()*/ )
                {
                    var af = new ArgField( obj, fi, arg );
                    args_dict[ af.Arg.Name ] = af;
                }
            }

            try
            {
                for ( int i = 0, len = args.Length - 1; i <= len; i++ )
                {
                    var a = args[ i ];
                    if ( (a != null) && (1 < a.Length) && (a[ 0 ] == '-') )
                    {
                        var argName = a.Substring( 1 );

                        if ( args_dict.TryGetValue( argName, out var af ) )
                        {
                            var k        = i + 1;
                            var argValue = (k <= len) ? args[ k ] : null;

                            af.Set( argValue );
                            i++;
                        }
                        else if ( throwForUnknownParams )
                        {
                            throw (new ArgumentException( $"'{argName}' is not a valid parameter." ));
                        }
                    }
                }

                foreach ( ArgField af in args_dict.Values )
                {
                    af.Validate();
                }
            }
            catch ( Exception ex )
            {
                Console.Error.WriteLine( ex );
                Usage( args_dict.Values );
            }

            return (args_dict.Values);
        }

        private static void Usage( IEnumerable< ArgField > args, bool forceExit = true )
        {
            Console.Error.WriteLine( $"Usage: {Process.GetCurrentProcess().ProcessName} [parameters...]" );
            foreach ( var af in args )
            {
                Console.Error.WriteLine( $"\t[-{af.Arg.Name}: {af.Arg.Title}]" );
            }

            if ( forceExit )
            {
                Environment.Exit( -1 );
            }
        }
    }
}

