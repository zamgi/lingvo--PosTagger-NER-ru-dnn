using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Xml.Linq;

using Lingvo.PosTagger.Tokenizing;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Urls
{
    /// <summary>
    /// 
    /// </summary>
    internal struct url_tuple
    {
        public string      value;
        public int         startIndex;
        public int         length;
        public UrlTypeEnum type;
#if DEBUG
        public override string ToString() => ((value != null) ? $"'{value}' [{startIndex}:{length}]" : $"[{startIndex}:{length}]");
#endif
        [M(O.AggressiveInlining)] internal url_t to_url() => new url_t( in this );
        [M(O.AggressiveInlining)] unsafe internal url_struct_t to_url_struct( char* _base ) => new url_struct_t( _startPtr: _base + startIndex, length, type );
    }

    /// <summary>
    /// 
    /// </summary>
    public enum UrlTypeEnum : byte
    {
        Url,
        Email,
    }

    /// <summary>
    /// 
    /// </summary>
    public sealed class url_t
    {
        [M(O.AggressiveInlining)] internal url_t( in url_tuple t )
        {
            startIndex = t.startIndex;
            length     = t.length;
            value      = t.value;
            type       = t.type;
        }
        [M(O.AggressiveInlining)] internal url_t( int _startIndex, int _length, UrlTypeEnum _type )
        {
            startIndex = _startIndex;
            length     = _length;
            type       = _type;
        }
        [M(O.AggressiveInlining)] internal url_t( int _startIndex, int _length, UrlTypeEnum _type, string _value )
        {
            startIndex = _startIndex;
            length     = _length;
            type       = _type;
            value      = _value;
        }

        public int    startIndex { [M(O.AggressiveInlining)] get; }
        public int    length     { [M(O.AggressiveInlining)] get; }
        public string value      { [M(O.AggressiveInlining)] get; }
        public UrlTypeEnum type  { [M(O.AggressiveInlining)] get; }
#if DEBUG
        public override string ToString() => ((value != null) ? $"'{value}' [{startIndex}:{length}]" : $"[{startIndex}:{length}]");
#endif
    }

    /// <summary>
    /// 
    /// </summary>
    unsafe public struct url_struct_t
    {
        internal url_struct_t( char* _startPtr, int _length, UrlTypeEnum _type )
        {
            startPtr = _startPtr;
            length   = _length;
            type     = _type;
        }

        public char*       startPtr { [M(O.AggressiveInlining)] get; }
        public int         length   { [M(O.AggressiveInlining)] get; }
        public UrlTypeEnum type     { [M(O.AggressiveInlining)] get; }
#if DEBUG
        public override string ToString() => $"{StringsHelper.ToString( startPtr, length )}";
#endif
        [M(O.AggressiveInlining)] unsafe internal url_t to_url_without_value( char* _base ) => new url_t( _startIndex: (int) (startPtr - _base), length, type );
        [M(O.AggressiveInlining)] unsafe internal url_t to_url_with_value( char* _base ) => new url_t( _startIndex: (int) (startPtr - _base), length, type, new string( startPtr, 0, length ) );
    }

    /// <summary>
    /// 
    /// </summary>
    public class UrlDetectorModel
    {
        public UrlDetectorModel( string urlDetectorResourcesXmlFilename ) => Initialize( XDocument.Load( urlDetectorResourcesXmlFilename ) );
        public UrlDetectorModel( StreamReader urlDetectorResourcesXmlStreamReader ) => Initialize( XDocument.Load( urlDetectorResourcesXmlStreamReader ) );
        public UrlDetectorModel( IEnumerable< string > firstLevelDomains, IEnumerable< string > uriSchemes ) => Initialize( firstLevelDomains, uriSchemes );

        public void Initialize( XDocument xdoc )
        {
            var firstLevelDomains = from xe in xdoc.Root.Element( "first-level-domains" ).Elements()
                                    select xe.Value;
            var uriSchemes = from xe in xdoc.Root.Element( "uri-schemes" ).Elements()
                             select xe.Value;

            Initialize( firstLevelDomains, uriSchemes );
        }
        private void Initialize( IEnumerable< string > firstLevelDomains, IEnumerable< string > uriSchemes )
        {
            FirstLevelDomains          = firstLevelDomains.ToHashset_4Urls();
            FirstLevelDomainsMaxLength = FirstLevelDomains.GetItemMaxLength();

            URIschemes                = uriSchemes.ToHashsetWithReverseValues_4Urls();
            URIschemesMaxLength       = URIschemes.GetItemMaxLength();
        }

        public HashSet< string > FirstLevelDomains          { [M(O.AggressiveInlining)] get; private set; }
        public int               FirstLevelDomainsMaxLength { [M(O.AggressiveInlining)] get; private set; }
        public HashSet< string > URIschemes                 { [M(O.AggressiveInlining)] get; private set; }
        public int               URIschemesMaxLength        { [M(O.AggressiveInlining)] get; private set; }
    }

    /// <summary>
    /// 
    /// </summary>
    public class UrlDetectorConfig
    {
        public UrlDetectorConfig() { }
        public UrlDetectorConfig( string urlDetectorResourcesXmlFilename ) => Model = new UrlDetectorModel( urlDetectorResourcesXmlFilename );
        public UrlDetectorConfig( StreamReader urlDetectorResourcesXmlStreamReader ) => Model = new UrlDetectorModel( urlDetectorResourcesXmlStreamReader );

        public UrlDetectorModel Model { get; set; }
        public UrlDetector.UrlExtractModeEnum UrlExtractMode { get; set; }
    }

    /// <summary>
    /// 
    /// </summary>
    unsafe public sealed class UrlDetector : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        public enum UrlExtractModeEnum
        {
            ValueAndPosition,
            Position,           
        }

        #region [.private field's.]
        private const int DEFAULT_LIST_CAPACITY                              = 100;
        private const int ALLOCATEURL_BYFIRSTLEVELDOMAIN_MAXRECURSIONNESTING = 10;
        private readonly HashSet< string >    _FirstLevelDomains;
        private readonly int                  _FirstLevelDomainsMaxLength;
        private readonly HashSet< string >    _URIschemes;
        private readonly int                  _URIschemesMaxLength;
        private readonly bool                 _ExtractValue;
        private readonly List< url_t >        _Urls;
        private readonly List< url_struct_t > _Urlstructs;
        private readonly StringBuilder        _StringBuilder;
        private readonly char[]               _FirstLevelDomainBuffer; //buffer for first-level-domain (right) part of url
        private readonly GCHandle             _FirstLevelDomainBufferGCHandle;
        private char*                         _FldBufferPtrBase;
        private readonly char[]               _URIschemesBuffer;       //buffer for URI-schemes (left) part of url
        private readonly GCHandle             _URIschemesBufferGCHandle;
        private char*                         _UriSchBufferPtrBase;
        private url_tuple                     _Url;
        private readonly CharType*            _CTM;  //xlat.CHARTYPE_MAP
        private readonly char*                _UIM;  //xlat.UPPER_INVARIANT_MAP        
        private char*                         _BASE; //start pointer into text
        private char*                         _Ptr;  //current pointer into text
        #endregion

        #region [.ctor().]
        public UrlDetector( UrlDetectorConfig config )
        {
            _ExtractValue = (config.UrlExtractMode == UrlExtractModeEnum.ValueAndPosition);

            _FirstLevelDomains          = config.Model.FirstLevelDomains;
            _FirstLevelDomainsMaxLength = config.Model.FirstLevelDomainsMaxLength;

            _URIschemes                 = config.Model.URIschemes;
            _URIschemesMaxLength        = config.Model.URIschemesMaxLength;

            _Urls                       = new List< url_t >( DEFAULT_LIST_CAPACITY );            
            _StringBuilder              = new StringBuilder();
            _Url                        = new url_tuple();
            _Urlstructs                 = new List< url_struct_t >( DEFAULT_LIST_CAPACITY );
			
            _CTM = xlat_Unsafe.Inst._CHARTYPE_MAP;
            _UIM = xlat_Unsafe.Inst._UPPER_INVARIANT_MAP;

            //-1-
            _FirstLevelDomainBuffer         = new char[ _FirstLevelDomainsMaxLength + 1 ];
            _FirstLevelDomainBufferGCHandle = GCHandle.Alloc( _FirstLevelDomainBuffer, GCHandleType.Pinned );
            _FldBufferPtrBase               = (char*) _FirstLevelDomainBufferGCHandle.AddrOfPinnedObject().ToPointer();

            //-2-
            _URIschemesBuffer         = new char[ _URIschemesMaxLength + 1 ];
            _URIschemesBufferGCHandle = GCHandle.Alloc( _URIschemesBuffer, GCHandleType.Pinned );
            _UriSchBufferPtrBase      = (char*) _URIschemesBufferGCHandle.AddrOfPinnedObject().ToPointer();
        }

        ~UrlDetector() => DisposeNativeResources();
        public void Dispose()
        {
            DisposeNativeResources();
            GC.SuppressFinalize( this );
        }
        private void DisposeNativeResources()
        {
            if ( _FldBufferPtrBase != null )
            {
                _FirstLevelDomainBufferGCHandle.Free();
                _FldBufferPtrBase = null;
            }

            if ( _UriSchBufferPtrBase != null )
            {
                _URIschemesBufferGCHandle.Free();
                _UriSchBufferPtrBase = null;
            }
        }
        #endregion

        unsafe public List< url_t > AllocateUrls( string text )
        {
            _Urls.Clear();

            fixed ( char* _base = text )
            {
                _BASE = _base;

                for ( _Ptr = _BASE; *_Ptr != '\0'; _Ptr++ )
                {
                    switch ( *_Ptr )
                    {
                        //-dot-
                        case '.':
                            if ( TryAllocateUrl_ByWWW() )
                            {
                                _Urls.Add( _Url.to_url() );
                            }
                            else if ( TryAllocateUrl_ByFirstLevelDomain( ALLOCATEURL_BYFIRSTLEVELDOMAIN_MAXRECURSIONNESTING ) )
                            {
                                _Urls.Add( _Url.to_url() );
                            }
                        break;

                        //-colon-
                        case ':':
//#if DEBUG
//var xxx = new string( _Ptr - 25, 0, 100 );
//#endif
                            if ( TryAllocateUrl_ByURIschemes() )
                            {
                                _Urls.Add( _Url.to_url() );
                            }
                        break;
                    }
                }
            }

            return (_Urls);
        }

        unsafe public List< url_struct_t > AllocateUrls( char* _base, int _length )
        {
            _Urlstructs.Clear();

            _BASE = _base;

            for ( _Ptr = _BASE; (*_Ptr != '\0') && (0 < _length); _Ptr++, _length-- )
            {
                switch ( *_Ptr )
                {
                    //-dot-
                    case '.':
                        if ( TryAllocateUrl_ByWWW() )
                        {
                            _Urlstructs.Add( _Url.to_url_struct( _base ) );
                        }
                        else if ( TryAllocateUrl_ByFirstLevelDomain( ALLOCATEURL_BYFIRSTLEVELDOMAIN_MAXRECURSIONNESTING ) )
                        {
                            _Urlstructs.Add( _Url.to_url_struct( _base ) );
                        }
                    break;

                    //-colon-
                    case ':':
//#if DEBUG
//var xxx = new string( _Ptr - 25, 0, 100 );
//#endif
                        if ( TryAllocateUrl_ByURIschemes() )
                        {
                            _Urlstructs.Add( _Url.to_url_struct( _base ) );
                        }
                    break;
                }
            }

            return (_Urlstructs);
        }

        /// <summary>
        /// 
        /// </summary>
        private bool TryAllocateUrl_ByWWW()
        {
            const int WWW_LENGTH = 3;

            #region [.check WWW on the left.]
            if ( _Ptr - WWW_LENGTH < _BASE )
            {
                return (false);
            }
            var isWWW = (*(_UIM + *(_Ptr - 1)) == 'W') &&
                        (*(_UIM + *(_Ptr - 2)) == 'W') &&
                        (*(_UIM + *(_Ptr - 3)) == 'W');
            if ( !isWWW )
            {
                return (false);
            }
            #endregion

            #region [.find-url-end-on-the-right.]
            var right_len = FindUrlEndOnTheRight( 0 );
            #endregion

            #region [.create url_t.]
            var left_ptr = _Ptr - WWW_LENGTH;
//#if DEBUG
//var xxx = new string( left_ptr - 25, 0, 75 );
//#endif
            var length = WWW_LENGTH + 1 + right_len;
            _Url.startIndex = (int) (left_ptr - _BASE);
            _Url.length     = length;
            _Url.type       = UrlTypeEnum.Url;
            if ( _ExtractValue )
            {
                _Url.value = new string( left_ptr, 0, length );
            }
            _Ptr += 1 + right_len;
            return (true);
            #endregion
        }

        /// <summary>
        /// 
        /// </summary>
        private bool TryAllocateUrl_ByFirstLevelDomain( int maxRecursionNesting )
        {
            if ( maxRecursionNesting <= 0 )
            {
                return (false);    
            }

            #region [.check first-level-domain on the right.]
            char ch;
            var right_len = 0;
            for ( _Ptr++; ; right_len++ )
            {
                ch = _Ptr[ right_len ];
                //char - '\0' - not marked as CharType.IsLetter
                #region
                /*if ( ch == '\0' )
                {
                    break;
                }
                */
                #endregion
                if ( (_CTM[ ch ] & CharType.IsLetter) != CharType.IsLetter )
                {
                    break;
                }

                if ( _FirstLevelDomainsMaxLength < right_len )
                {
                    return (false);
                }

                //to upper
                _FldBufferPtrBase[ right_len ] = _UIM[ ch ];
            }

            if ( right_len == 0 )
            {
                return (false);
            }

            _StringBuilder.Clear().Append( _FirstLevelDomainBuffer, 0, right_len );
            if ( !_FirstLevelDomains.Contains( _StringBuilder.ToString() ) )
            {
                return (false);
            }
            #endregion

            #region [.find-url-end-on-the-right.]
            if ( xlat.IsDot( ch ) )
            {
//#if DEBUG
//var xxx1 = new string( _Ptr - 25, 0, 75 );
//#endif
                var safe_Ptr = _Ptr;
                _Ptr += right_len;
                if ( TryAllocateUrl_ByFirstLevelDomain( maxRecursionNesting-- ) )
                {
                    return (true);
                }
                _Ptr = safe_Ptr;
            }

            _Ptr--;
            if ( xlat.IsURIschemesPathSeparator( ch ) )
            {
                right_len = FindUrlEndOnTheRight( right_len );
            }
            #endregion

            #region [.find-url-end-on-the-left.]
            var left_len = FindUrlEndOnTheLeft( 1, out var urlType );
            //skip url with empty left-part
            if ( left_len == 0 )
            {
                return (false);
            }
            #endregion

            #region [.create url_t.]
            var left_ptr = _Ptr - left_len;
//#if DEBUG
//var xxx = new string( left_ptr - 25, 0, 75 );
//#endif
            var length = left_len + 1 + right_len;
            _Url.startIndex = (int) (left_ptr - _BASE);
            _Url.length     = length;
            _Url.type       = urlType; //---(IsSeemsLikeEmail( left_ptr, length ) ? UrlTypeEnum.Email : UrlTypeEnum.Url);
            if ( _ExtractValue )
            {
                _Url.value = new string( left_ptr, 0, length );
            }
            _Ptr += 1 + right_len;
            return (true);
            #endregion
        }

        /// <summary>
        /// 
        /// </summary>
        private bool TryAllocateUrl_ByURIschemes()
        {
            #region [.check URI-schemes on the left.]
            var left_len = 0;
            for ( ; ; left_len++ )
            {
                var p = _Ptr - left_len - 1;
                if ( p < _BASE )
                {                    
                    break;
                }

                var ch = *p;
                if ( (_CTM[ ch ] & CharType.IsURIschemesChar) != CharType.IsURIschemesChar )
                {
                    break;
                }

                if ( _URIschemesMaxLength < left_len )
                {
                    return (false);
                }

                //to upper
                _UriSchBufferPtrBase[ left_len ] = _UIM[ ch ];
            }

            if ( left_len == 0 )
            {
                return (false);
            }

            _StringBuilder.Clear().Append( _URIschemesBuffer, 0, left_len );
            if ( !_URIschemes.Contains( _StringBuilder.ToString() ) )
            {
                _Ptr++;
                return (false);
            }
            #endregion

            #region [.find-url-end-on-the-right.]
            var right_len = FindUrlEndOnTheRight( 0 );
            #endregion

            #region [.create url_t.]
            var left_ptr = _Ptr - left_len;
//#if DEBUG
//var xxx = new string( left_ptr - 25, 0, 75 );
//#endif
            var length = left_len + 1 + right_len;
            _Url.startIndex = (int) (left_ptr - _BASE);
            _Url.length     = length;
            _Url.type       = IsSeemsLikeEmail( left_ptr, length ) ? UrlTypeEnum.Email : UrlTypeEnum.Url;
            if ( _ExtractValue )
            {
                _Url.value = new string( left_ptr, 0, length );
            }
            _Ptr += 1 + right_len;
            return (true);
            #endregion
        }

        /// <summary>
        /// 
        /// </summary>
        [M(O.AggressiveInlining)] private int FindUrlEndOnTheRight( int offsetToRight )
        {
            var right_len = offsetToRight;
            for ( ; ; right_len++ )
            {
                var ch = _Ptr[ right_len ];
                //char - '\0' - marked as CharType.IsUrlBreak
                #region comm.
                /*if ( ch == '\0' )
                {
                    right_len--;
                    break;
                }*/
                #endregion

                if ( (_CTM[ ch ] & CharType.IsUrlBreak) == CharType.IsUrlBreak )
                {
                    for ( right_len--; 0 <= right_len; right_len-- )
                    {
                        ch = _Ptr[ right_len ];
                        if ( ch == '/' )
                            break;
                        if ( (_CTM[ ch ] & CharType.IsPunctuation) != CharType.IsPunctuation )
                            break;
                    }
                    break;

                    #region commented
                    /*
                    right_len--;
                    #region [.if ends with dot.]
                    ch = _Ptr[ right_len ];
                    if ( xlat.IsSentEndChar( ch ) )
                        right_len--;
                    #endregion
                    break;
                    */ 
                    #endregion
                }
            }
            return ((right_len > 0) ? right_len : 0);
        }
        /// <summary>
        /// 
        /// </summary>
        [M(O.AggressiveInlining)] private int FindUrlEndOnTheLeft( int offsetToLeft, out UrlTypeEnum urlType )
        {
            var left_len = offsetToLeft;
            urlType = UrlTypeEnum.Url;
            for ( ; ; left_len++ )
            {
                var p = _Ptr - left_len;
                if ( p <= _BASE )
                {
                    while ( p < _BASE )
                    {
                        p++;
                        left_len--;
                    }

                    for ( /*left_len--*/; 0 <= left_len; left_len-- )
                    {
                        var ch = *(_Ptr - left_len);
                        if ( ch == '/' )
                            break;
                        var ct = _CTM[ ch ];
                        if ( (ct & CharType.IsWhiteSpace) == CharType.IsWhiteSpace )
                            continue;
                        if ( (ct & CharType.IsPunctuation) != CharType.IsPunctuation )
                        {
                            if ( IsEmail( left_len ) )
                            {
                                urlType = UrlTypeEnum.Email;
                            }
                            break;
                        }
                    }
                    break;
                }

                if ( (_CTM[ *p ] & CharType.IsUrlBreak) == CharType.IsUrlBreak )
                {
                    for ( left_len--; 0 <= left_len; left_len-- )
                    {
                        var ch = *(_Ptr - left_len);
                        if ( ch == '/' )
                            break;
                        var ct = _CTM[ ch ];
                        if ( (ct & CharType.IsWhiteSpace) == CharType.IsWhiteSpace )
                            continue;
                        if ( (ct & CharType.IsPunctuation) != CharType.IsPunctuation )
                        {
                            if ( IsEmail( left_len ) )
                            {
                                urlType = UrlTypeEnum.Email;
                            }
                            break;
                        }
                    }
                    break;
                }
            }
            return ((left_len > 0) ? left_len : 0);
        }        

        [M(O.AggressiveInlining)] private bool IsEmail( int startIndex )
        {
            for ( ; 0 <= startIndex; startIndex-- )
            {
                var ptr = (_Ptr - startIndex);
                var ch  = *ptr;
                switch ( ch )
                {
                    case '(':
                        if ( Find_AT_OnTheRight_1( ptr ) )
                        {
                            return (true);
                        }
                    break;
                    case '[':
                        if ( Find_AT_OnTheRight_2( ptr ) )
                        {
                            return (true);
                        }
                    break;
                    case '@':
                        return (true);
                }
            }
            return (false);
        }
        [M(O.AggressiveInlining)] private bool Find_AT_OnTheRight_1( char* ptr )
        {
            #region [.check @ on the right.]
            var isAT = (*(_UIM + *(ptr + 1)) == 'A') &&
                       (*(_UIM + *(ptr + 2)) == 'T') &&
                       (*(_UIM + *(ptr + 3)) == ')');
            return (isAT);
            #endregion
        }
        [M(O.AggressiveInlining)] private bool Find_AT_OnTheRight_2( char* ptr )
        {
            #region [.check @ on the right.]
            var isAT = (*(_UIM + *(ptr + 1)) == 'A') &&
                       (*(_UIM + *(ptr + 2)) == 'T') &&
                       (*(_UIM + *(ptr + 3)) == ']');
            return (isAT);
            #endregion
        }

        [M(O.AggressiveInlining)] private static bool IsSeemsLikeEmail( char* ptr, int length )
        {
            var has_dog = false;
            for ( length--; 0 <= length; length-- )
            {
                var ch = ptr[ length ];
                if ( xlat.IsSlash( ch ) )
                {
                    return (false);
                }
                has_dog |= (ch == '@');
            }
            return (has_dog);
        }
    }
}

namespace Lingvo.PosTagger.Urls
{
    /// <summary>
    /// 
    /// </summary>
    internal static class UrlDetectorExtensions
    {
        public static HashSet< string > ToHashset_4Urls( this IEnumerable< string > seq )
            => new HashSet< string >( seq.Select( d => (d != null) ? d.Trim().ToUpperInvariant() : null ).Where( d => !string.IsNullOrEmpty( d ) ) );
        public static HashSet< string > ToHashsetWithReverseValues_4Urls( this IEnumerable< string > seq )
            => new HashSet< string >( seq.Select( d => (d != null) ? new string( d.Trim().Reverse().ToArray() ).ToUpperInvariant() : null ).Where( d => !string.IsNullOrEmpty( d ) ) );
        public static int GetItemMaxLength( this HashSet< string > hs ) => ((hs.Count != 0) ? hs.Max( d => d.Length ) : 0);        
    }
}
