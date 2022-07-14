using System.Text;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    unsafe public sealed class UmlautesNormalizer
    {
        private StringBuilder _Buff;
        public UmlautesNormalizer() => _Buff = new StringBuilder( 100 );

        [M(O.AggressiveInlining)] public string Normalize( string value )
        {            
            fixed ( char* _base = value )
            {
                return (Normalize( _base, value.Length ));
            }
        }
        [M(O.AggressiveInlining)] public string Normalize( char* _base, int length )
        {            
            for ( int i = 0; i < length; i++ )
            {
                var ch = *(_base + i);
                switch ( ch )
                {
                    case 'ä': _Buff.Append( 'a' ).Append( 'e' ); break;
                    case 'ö': _Buff.Append( 'o' ).Append( 'e' ); break;
                    case 'ü': _Buff.Append( 'u' ).Append( 'e' ); break;
                    case 'Ä': _Buff.Append( 'A' ).Append( 'e' ); break; //.Append( 'E' ); break;
                    case 'Ö': _Buff.Append( 'O' ).Append( 'e' ); break; //.Append( 'E' ); break;
                    case 'Ü': _Buff.Append( 'U' ).Append( 'e' ); break; //.Append( 'E' ); break;
                    case 'ß': _Buff.Append( 's' ).Append( 's' ); break;
                    default : _Buff.Append( ch ); break;
                }
            }
            var value = _Buff.ToString(); _Buff.Clear();
            return (value);
        }
        [M(O.AggressiveInlining)] public string Normalize_ToUpper( string value )
        {            
            fixed ( char* _base = value )
            {
                return (Normalize_ToUpper( _base, value.Length ));
            }
        }
        [M(O.AggressiveInlining)] public string Normalize_ToUpper( char* _base, int length )
        {            
            for ( int i = 0; i < length; i++ )
            {
                var ch = *(_base + i);
                switch ( ch )
                {
                    case 'ä': _Buff.Append( 'A' ).Append( 'E' ); break;
                    case 'ö': _Buff.Append( 'O' ).Append( 'E' ); break;
                    case 'ü': _Buff.Append( 'U' ).Append( 'E' ); break;
                    case 'Ä': _Buff.Append( 'A' ).Append( 'E' ); break; //.Append( 'E' ); break;
                    case 'Ö': _Buff.Append( 'O' ).Append( 'E' ); break; //.Append( 'E' ); break;
                    case 'Ü': _Buff.Append( 'U' ).Append( 'E' ); break; //.Append( 'E' ); break;
                    case 'ß': _Buff.Append( 'S' ).Append( 'S' ); break;
                    default : _Buff.Append( ch ); break;
                }
            }
            var value = _Buff.ToString(); _Buff.Clear();
            return (value);
        }

        [M(O.AggressiveInlining)] public static bool IsUmlauteSymbol( char ch )
		{
            switch ( ch )
            {
                case 'ä':
                case 'ö':
                case 'ü':
                case 'Ä':
                case 'Ö':
                case 'Ü':
                case 'ß':
                    return (true);
            }
			return (false);
		}
    }
}