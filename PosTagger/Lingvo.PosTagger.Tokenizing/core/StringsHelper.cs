using System;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    unsafe public static class StringsHelper
    {
        private static char*     _UPPER_INVARIANT_MAP;
        private static CharType* _CTM;
        static StringsHelper()
        {
            _UPPER_INVARIANT_MAP = xlat_Unsafe.Inst._UPPER_INVARIANT_MAP;
            _CTM                 = xlat_Unsafe.Inst._CHARTYPE_MAP;
        }

        [M(O.AggressiveInlining)] public static string ToUpperInvariant( string value )
        {
            var len = value.Length;
            if ( 0 < len )
            {
                var valueUpper = new string( value ); //---string.Copy( value );
                fixed ( char* valueUpper_ptr = valueUpper )
                {
                    for ( int i = 0; i < len; i++ )
                    {
                        var ptr = valueUpper_ptr + i;
                        *ptr = *(_UPPER_INVARIANT_MAP + *ptr);
                    }
                }
                return (valueUpper);
            }
            return (string.Empty);
        }
        [M(O.AggressiveInlining)] public static void   ToUpperInvariant( char* wordFrom, char* bufferTo )
        {
            for ( ; ; wordFrom++, bufferTo++ )
            {
                var ch = *wordFrom;
                *bufferTo = *(_UPPER_INVARIANT_MAP + ch);
                if ( ch == '\0' )
                    return;
            }            
        }
        [M(O.AggressiveInlining)] public static void   ToUpperInvariantInPlace( string value )
        {
            fixed ( char* value_ptr = value )
            {
                ToUpperInvariantInPlace( value_ptr );
            }
        }
        [M(O.AggressiveInlining)] public static void   ToUpperInvariantInPlace( char* word )
        {
            for ( ; ; word++ )
            {
                var ch = *word;
                if ( ch == '\0' )
                    return;
                *word = *(_UPPER_INVARIANT_MAP + ch);
            }
        }
        [M(O.AggressiveInlining)] public static void   ToUpperInvariantInPlace( char* word, int length )
        {
            for ( length--; 0 <= length; length-- )
            {
                word[ length ] = _UPPER_INVARIANT_MAP[ word[ length ] ];
            }
        }
        [M(O.AggressiveInlining)] public static string ToUpperInvariantInPlace_2( string value )
        {
            fixed ( char* value_ptr = value )
            {
                ToUpperInvariantInPlace( value_ptr );
            }
            return (value);
        }

        [M(O.AggressiveInlining)] public static void   ToUpperInvariantInPlaceFirstLetter( string value )
        {
            fixed ( char* value_ptr = value )
            {
                var ch = *value_ptr;
                if ( ch != '\0' )
                {
                    *value_ptr = _UPPER_INVARIANT_MAP[ ch ];
                }
            }
        }
        [M(O.AggressiveInlining)] public static void   ToUpperInvariantInPlaceFirstLetter( char* word )
        {
            var ch = *word;
            if ( ch != '\0' )
            {
                *word = _UPPER_INVARIANT_MAP[ ch ];
            }
        }

        [M(O.AggressiveInlining)] public static string ToLowerInvariant( string value ) => value.ToLowerInvariant();

        [M(O.AggressiveInlining)] public static bool IsEqual( string first, string second )
        {
            int length = first.Length;
            if ( length != second.Length )
            {
                return (false);
            }
            if ( length == 0 )
            {
                return (true);
            }

            fixed ( char* first_ptr  = first )
            fixed ( char* second_ptr = second )
            {
                for ( int i = 0; i < length; i++ )
                {
                    if ( *(first_ptr + i) != *(second_ptr + i) ) //if ( GetLetter( first, i ) != GetLetter( second, i ) )
                    {
                        return (false);
                    }
                }
            }
            return (true);
        }
        [M(O.AggressiveInlining)] public static bool IsEqual( string first, char* second_ptr, int secondLength )
        {
            /*
            if ( first.Length != secondLength )
            {
                return (false);
            }
            if ( secondLength == 0 )
            {
                return (true);
            }
            */

            fixed ( char* first_ptr  = first )
            {
                for ( int i = 0; i < secondLength; i++ )
                {
                    if ( *(first_ptr + i) != *(second_ptr + i) )
                    {
                        return (false);
                    }
                }
            }
            return (true);
        }

        [M(O.AggressiveInlining)] public static bool IsEqual( string first, int firstIndex, string second )
        {
            int length = first.Length - firstIndex;
            if ( length != second.Length )
            {
                return (false);
            }
            if ( length == 0 )
            {
                return (true);
            }

            fixed ( char* first_base = first  )
            fixed ( char* second_ptr = second )
            {
                char* first_ptr = first_base + firstIndex;
                for ( int i = 0; i < length; i++ )
                {
                    if ( *(first_ptr + i) != *(second_ptr + i) ) //if ( GetLetter( first, i ) != GetLetter( second, i ) )
                    {
                        return (false);
                    }
                }
            }
            return (true);
        }
        [M(O.AggressiveInlining)] public static bool IsEqual( string first, int firstIndex, char* second_ptr, int secondLength )
        {
            int length = first.Length - firstIndex;
            if ( length != secondLength )
            {
                return (false);
            }
            if ( secondLength == 0 )
            {
                return (true);
            }

            fixed ( char* first_base = first  )
            {
                char* first_ptr = first_base + firstIndex;
                for ( int i = 0; i < secondLength; i++ )
                {
                    if ( *(first_ptr + i) != *(second_ptr + i) ) //if ( GetLetter( first, i ) != GetLetter( second, i ) )
                    {
                        return (false);
                    }
                }
            }
            return (true);
        }

        [M(O.AggressiveInlining)] public static bool IsEqual( IntPtr x, IntPtr y )
        {
            if ( x == y )
                return (true);

            for ( char* x_ptr = (char*) x.ToPointer(),
                        y_ptr = (char*) y.ToPointer(); ; x_ptr++, y_ptr++ )
            {
                var x_ch = *x_ptr;

                if ( x_ch != *y_ptr )
                    return (false);
                if ( x_ch == '\0' )
                    return (true);
            }
        }
        [M(O.AggressiveInlining)] public static bool IsEqual( char* x, char* y )
        {
            if ( x == y )
                return (true);

            for ( ; ; x++, y++)
            {
                var x_ch = *x;

                if ( x_ch != *y )
                    return (false);
                if ( x_ch == '\0' )
                    return (true);
            }
        }

        [M(O.AggressiveInlining)] public static int GetLength( char* _base )
        {
            for ( var ptr = _base; ; ptr++ )
            {
                if ( *ptr == '\0' )
                {
                    return ((int)(ptr - _base));
                }
            }
        }
        [M(O.AggressiveInlining)] public static int GetLength( IntPtr _base ) => GetLength( (char*) _base );

        [M(O.AggressiveInlining)] public static string ToString( char* value )
        {
            if ( value == null )
            {
                return (null);
            }

            var length = GetLength( value );
            if ( length == 0 )
            {
                return (string.Empty);
            }

            var str = new string( '\0', length );
            fixed ( char* str_ptr = str )
            {                
                for ( var wf_ptr = str_ptr; ; )
                {
                    var ch = *(value++);
                    if ( ch == '\0' )
                        break;
                    *(wf_ptr++) = ch;
                }
            }
            return (str);
        }
        [M(O.AggressiveInlining)] public static string ToString( char* value, int length )
        {
            if ( value == null )
            {
                return (null);
            }

            if ( length == 0 )
            {
                return (string.Empty);
            }

            var str = new string( '\0', length );
            fixed ( char* str_ptr = str )
            {
                for ( var wf_ptr = str_ptr; 0 < length; length-- )
                {
                    var ch = *(value++);
                    if ( ch == '\0' )
                        break;
                    *(wf_ptr++) = ch;
                }
            }
            return (str);
        }
        [M(O.AggressiveInlining)] public static string ToString( IntPtr value ) => ToString( (char*) value );

        [M(O.AggressiveInlining)] public static CharType GetCharTypes( string value )
        {
            fixed ( char* value_ptr = value )
            {
                return (GetCharTypes( value_ptr ));
            }
        }
        [M(O.AggressiveInlining)] public static CharType GetCharTypes( char* value )
        {
            var ct = CharType.__UNDEFINE__;
            for ( ; ; value++ )
            {
                var ch = *value;
                if ( ch == '\0' )
                {
                    return (ct);
                }
                ct |= _CTM[ ch ];
            }
        }

        [M(O.AggressiveInlining)] public static (bool OnlyLetters, bool WhiteSpace) ContainsOnlyLettersOrWhiteSpace( string value )
        {
            fixed ( char* value_ptr = value )
            {
                return (ContainsOnlyLettersOrWhiteSpace( value_ptr ));
            }
        }
        [M(O.AggressiveInlining)] public static (bool OnlyLetters, bool HasWhiteSpace) ContainsOnlyLettersOrWhiteSpace( char* value )
        {
            for ( ; ; value++ )
            {
                var ch = *value;
                if ( ch == '\0' )
                {
                    return (true, false);
                }

                if ( (_CTM[ ch ] & CharType.IsLetter) != CharType.IsLetter )
                {
                    if ( (_CTM[ ch ] & CharType.IsWhiteSpace) == CharType.IsWhiteSpace )
                    {
                        return (false, true);
                    }

                    for ( ; ch != '\0'; ch = *value++ )
                    {                   
                        if ( (_CTM[ ch ] & CharType.IsWhiteSpace) == CharType.IsWhiteSpace )
                        {
                            return (false, true);
                        }
                    }
                    return (false, false);
                }
            }
        }

        [M(O.AggressiveInlining)] public static (bool OnlyLetters, bool HasWhiteSpaceOrDot) ContainsOnlyLettersOrWhiteSpaceOrDot( string value )
        {
            fixed ( char* value_ptr = value )
            {
                return (ContainsOnlyLettersOrWhiteSpaceOrDot( value_ptr ));
            }
        }
        [M(O.AggressiveInlining)] public static (bool OnlyLetters, bool HasWhiteSpaceOrDot) ContainsOnlyLettersOrWhiteSpaceOrDot( char* value )
        {
            for ( ; ; value++ )
            {
                var ch = *value;
                if ( ch == '\0' )
                {
                    return (true, false);
                }

                if ( (_CTM[ ch ] & CharType.IsLetter) != CharType.IsLetter )
                {
                    //var hasWhiteSpace = false;
                    //var hasDot        = false;
                    //for ( ; ch != '\0'; ch = *value++ )
                    //{
                    //    if ( (_CTM[ ch ] & CharType.IsWhiteSpace) == CharType.IsWhiteSpace ) hasWhiteSpace = true;
                    //    else if ( xlat.IsDot( ch ) ) hasDot = true;
                    //}
                    //return (false, hasWhiteSpace, hasDot);

                    if ( xlat.IsDot( ch ) || (_CTM[ ch ] & CharType.IsWhiteSpace) == CharType.IsWhiteSpace )
                    {
                        return (false, true);
                    }

                    for ( ; ch != '\0'; ch = *value++ )
                    {
                        if ( xlat.IsDot( ch ) || (_CTM[ ch ] & CharType.IsWhiteSpace) == CharType.IsWhiteSpace )
                        {
                            return (false, true);
                        }
                    }
                    return (false, false);
                }
            }
        }

        [M(O.AggressiveInlining)] public static bool ContainsOnlyLetters( string value )
        {
            fixed ( char* value_ptr = value )
            {
                return (ContainsOnlyLetters( value_ptr ));
            }
        }
        [M(O.AggressiveInlining)] public static bool ContainsOnlyLetters( char* value )
        {
            for ( ; ; value++ )
            {
                var ch = *value;
                if ( ch == '\0' )
                {
                    return (true);
                }
                if ( (_CTM[ ch ] & CharType.IsLetter) != CharType.IsLetter )
                {
                    return (false);
                }
            }
        }

        /*[M(O.AggressiveInlining)] public static bool ContainsWhiteSpace( string value )
        {
            fixed ( char* value_ptr = value )
            {
                return (ContainsWhiteSpace( value_ptr ));
            }
        }
        [M(O.AggressiveInlining)] public static bool ContainsWhiteSpace( char* value )
        {
            for ( ; ; value++ )
            {
                var ch = *value;
                if ( ch == '\0' )
                {
                    return (false);
                }
                if ( (_CTM[ ch ] & CharType.IsWhiteSpace) == CharType.IsWhiteSpace )
                {
                    return (true);
                }
            }
        }*/

        [M(O.AggressiveInlining)] public static bool ContainsPunctuation( string value )
        {
            fixed ( char* value_ptr = value )
            {
                return (ContainsPunctuation( value_ptr ));
            }
        }
        [M(O.AggressiveInlining)] public static bool ContainsPunctuation( char* value )
        {
            for ( ; ; value++ )
            {
                var ch = *value;
                if ( ch == '\0' )
                {
                    return (false);
                }

                if ( (_CTM[ ch ] & CharType.IsPunctuation) == CharType.IsPunctuation )
                {
                    return (true);
                }
            }
        }

        [M(O.AggressiveInlining)] public static (bool OnlyLetters, bool HasHyphenOrDot) ContainsOnlyLettersOrHyphenOrDot( string value )
        {
            fixed ( char* value_ptr = value )
            {
                return (ContainsOnlyLettersOrHyphenOrDot( value_ptr ));
            }
        }
        [M(O.AggressiveInlining)] public static (bool OnlyLetters, bool HasHyphenOrDot) ContainsOnlyLettersOrHyphenOrDot( char* value )
        {
            for ( ; ; value++ )
            {
                var ch = *value;
                if ( ch == '\0' )
                {
                    return (true, false);
                }

                if ( (_CTM[ ch ] & CharType.IsLetter) != CharType.IsLetter )
                {
                    if ( xlat.IsDot( ch ) || (_CTM[ ch ] & CharType.IsHyphen) == CharType.IsHyphen )
                    {
                        return (false, true);
                    }

                    for ( ; ch != '\0'; ch = *value++ )
                    {
                        if ( xlat.IsDot( ch ) || (_CTM[ ch ] & CharType.IsHyphen) == CharType.IsHyphen )
                        {
                            return (false, true);
                        }
                    }
                    return (false, false);
                }
            }
        }

        [M(O.AggressiveInlining)] public static void ReplaceInPlace( string value, char oldChar, char newChar )
        {
            fixed ( char* value_ptr = value )
            {
                ReplaceInPlace( value_ptr, oldChar, newChar );
            }
        }
        [M(O.AggressiveInlining)] public static void ReplaceInPlace( char* word, char oldChar, char newChar )
        {
            for ( ; ; word++ )
            {
                var ch = *word;
                if ( ch == '\0' )
                    return;
                if ( ch == oldChar )
                    *word = newChar;
            }
        }
    }
}
