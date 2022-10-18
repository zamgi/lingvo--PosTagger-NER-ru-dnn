using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

using Lingvo.PosTagger.SentSplitting;
using Lingvo.PosTagger.Urls;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    unsafe sealed public class Tokenizer : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        public delegate void ProcessSentCallbackDelegate( List< word_t > words );

        /// <summary>
        /// 
        /// </summary>
        [Flags] private enum SpecialCharType : byte
        {
            __UNDEFINE__                = 0x0,

            InterpreteAsWhitespace      = 0x1,
            BetweenLetterOrDigit        = (1 << 1),
            BetweenDigit                = (1 << 2),
            TokenizeDifferentSeparately = (1 << 3),
            DotChar                     = (1 << 4),
        }

        /// <summary>
        /// 
        /// </summary>
        unsafe private sealed class UnsafeConst
        {
            #region [.static & xlat table's.]
            public  static readonly char*  MAX_PTR                          = (char*) (0xffffffffFFFFFFFF);
            private const string           INCLUDE_INTERPRETE_AS_WHITESPACE = "¤¦§¶"; //"¥©¤¦§®¶€™<>";
            private const char             DOT                              = '\u002E'; /* 0x2E, 46, '.' */
            #endregion

            public readonly SpecialCharType* _SPEC_CHARTYPE_MAP;
            private UnsafeConst()
            {
                #region [.xlat table's.]
                #region comm.
                //var BETWEEN_LETTER_OR_DIGIT_RU    = new[]
                //                                    {
                //                                    '\u0026', /* 0x26  , 38  , '&' */
                //                                    '\u0027', /* 0x27  , 39  , ''' */
                //                                    '\u002D', /* 0x2D  , 45  , '-' */
                //                                    '\u005F', /* 0x5F  , 95  , '_' */
                //                                    '\u00AD', /* 0xAD  , 173 , '­' */
                //                                    '\u055A', /* 0x55A , 1370, '՚' */
                //                                    '\u055B', /* 0x55B , 1371, '՛' */
                //                                    '\u055D', /* 0x55D , 1373, '՝' */
                //                                    '\u2012', /* 0x2012, 8210, '‒' */
                //                                    '\u2013', /* 0x2013, 8211, '–' */
                //                                    '\u2014', /* 0x2014, 8212, '—' */
                //                                    '\u2015', /* 0x2015, 8213, '―' */
                //                                    '\u2018', /* 0x2018, 8216, '‘' */
                //                                    '\u2019', /* 0x2019, 8217, '’' */
                //                                    '\u201B', /* 0x201B, 8219, '‛' */
                //                                    };
                #endregion
                var BETWEEN_LETTER_OR_DIGIT_EN    = new[] 
                                                    { 
                                                    '\u0026', /* 0x26  , 38  , '&' */
                                                    //'\u0027', /* 0x27  , 39  , ''' */
                                                    '\u002D', /* 0x2D  , 45  , '-' */
                                                    '\u005F', /* 0x5F  , 95  , '_' */
                                                    '\u00AD', /* 0xAD  , 173 , '­' */
                                                    //'\u055A', /* 0x55A , 1370, '՚' */
                                                    //'\u055B', /* 0x55B , 1371, '՛' */
                                                    //'\u055D', /* 0x55D , 1373, '՝' */
                                                    '\u2012', /* 0x2012, 8210, '‒' */
                                                    '\u2013', /* 0x2013, 8211, '–' */
                                                    '\u2014', /* 0x2014, 8212, '—' */
                                                    '\u2015', /* 0x2015, 8213, '―' */
                                                    '\u2018', /* 0x2018, 8216, '‘' */
                                                    //'\u2019', /* 0x2019, 8217, '’' */
                                                    '\u201B', /* 0x201B, 8219, '‛' */
                                                    };
                var BETWEEN_DIGIT                 = new[] 
                                                    { 
                                                    '\u0022', /* 0x22   , 34   , '"'  */
                                                    '\u002C', /* 0x2C   , 44   , ','  */
                                                    '\u003A', /* 0x3A   , 58   , ':'  */
                                                    '\u3003', /* 0x3003 , 12291, '〃' */
                                                    //-ERROR-!!!-DOT, /* и  0x2E   , 46   , '.' - хотя это и так работает */
                                                    };
                var TOKENIZE_DIFFERENT_SEPARATELY = new[] 
                                                    {             
                                                    '\u2012', /* 0x2012 , 8210 , '‒' */
                                                    '\u2013', /* 0x2013 , 8211 , '–' */
                                                    '\u2014', /* 0x2014 , 8212 , '—' */
                                                    '\u2015', /* 0x2015 , 8213 , '―' */
                                                    '\u2018', /* 0x2018 , 8216 , '‘' */
                                                    '\u2019', /* 0x2019 , 8217 , '’' */
                                                    '\u201B', /* 0x201B , 8219 , '‛' */
                                                    '\u201C', /* 0x201C , 8220 , '“' */
                                                    '\u201D', /* 0x201D , 8221 , '”' */
                                                    '\u201E', /* 0x201E , 8222 , '„' */
                                                    '\u201F', /* 0x201F , 8223 , '‟' */
                                                    '\u2026', /* 0x2026 , 8230 , '…' */
                                                    '\u0021', /* 0x21   , 33   , '!' */
                                                    '\u0022', /* 0x22   , 34   , '"' */
                                                    '\u0026', /* 0x26   , 38   , '&' */
                                                    '\u0027', /* 0x27   , 39   , ''' */
                                                    '\u0028', /* 0x28   , 40   , '(' */
                                                    '\u0029', /* 0x29   , 41   , ')' */
                                                    '\u002C', /* 0x2C   , 44   , ',' */
                                                    '\u002D', /* 0x2D   , 45   , '-' */
                                                    //DOT, //'\u002E', /* 0x2E   , 46   , '.' */
                                                    '\u3003', /* 0x3003 , 12291, '〃' */
                                                    '\u003A', /* 0x3A   , 58   , ':' */
                                                    '\u003B', /* 0x3B   , 59   , ';' */
                                                    '\u003F', /* 0x3F   , 63   , '?' */
                                                    '\u055A', /* 0x55A  , 1370 , '՚' */
                                                    '\u055B', /* 0x55B  , 1371 , '՛'  */
                                                    '\u055D', /* 0x55D  , 1373 , '՝' */
                                                    '\u005B', /* 0x5B   , 91   , '[' */
                                                    '\u005D', /* 0x5D   , 93   , ']' */
                                                    '\u005F', /* 0x5F   , 95   , '_' */
                                                    '\u05F4', /* 0x5F4  , 1524 , '״' */
                                                    '\u007B', /* 0x7B   , 123  , '{' */
                                                    '\u007D', /* 0x7D   , 125  , '}' */
                                                    '\u00A1', /* 0xA1   , 161  , '¡' */
                                                    '\u00AB', /* 0xAB   , 171  , '«' */
                                                    '\u00AD', /* 0xAD   , 173  , '­' */
                                                    '\u00BB', /* 0xBB   , 187  , '»' */
                                                    '\u00BF', /* 0xBF   , 191  , '¿' */
                                                    '/',
                                                    '¥', '©', '®', '€', '™', '°', '№', '$', '%',
                                                    '<', '>',
                                                    };
                #endregion

                //-1-//
                var spec_chartype_map = new byte/*SpecialCharType*/[ char.MaxValue + 1 ];
                fixed ( /*SpecialCharType*/byte* cctm = spec_chartype_map )        
                {
                    for ( var c = char.MinValue; /*c <= char.MaxValue*/; c++ )
                    {
                        if ( /*char.IsWhiteSpace( c ) ||*/ char.IsPunctuation( c ) )
                        {
                            *(cctm + c) = (byte) SpecialCharType.InterpreteAsWhitespace;
                        }

                        if ( c == char.MaxValue )
                        {
                            break;
                        }
                    }

                    foreach ( var c in INCLUDE_INTERPRETE_AS_WHITESPACE )
                    {
                        *(cctm + c) = (byte) SpecialCharType.InterpreteAsWhitespace;
                    }

                    foreach ( var c in TOKENIZE_DIFFERENT_SEPARATELY )
                    {
                        *(cctm + c) = (byte) SpecialCharType.TokenizeDifferentSeparately;
                    }

                    //var between_letter_or_digit = (languageType == LanguageTypeEnum.En) ? BETWEEN_LETTER_OR_DIGIT_EN : BETWEEN_LETTER_OR_DIGIT_RU;
                    foreach ( var c in BETWEEN_LETTER_OR_DIGIT_EN )
                    {
                        *(cctm + c) |= (byte) SpecialCharType.BetweenLetterOrDigit;
                    }

                    foreach ( var c in BETWEEN_DIGIT )
                    {
                        *(cctm + c) |= (byte) SpecialCharType.BetweenDigit;
                    }

                    //-ERROR-!!!-*(cctm + DOT) |= (byte) SpecialCharType.DotChar;
                    //-ONLY-SO--!!!-
                    *(cctm + DOT) = (byte) SpecialCharType.DotChar;
                }

                var spec_chartype_map_GCHandle = GCHandle.Alloc( spec_chartype_map, GCHandleType.Pinned );
                _SPEC_CHARTYPE_MAP = (SpecialCharType*) spec_chartype_map_GCHandle.AddrOfPinnedObject().ToPointer();
            }
            public static UnsafeConst Inst { get; } = new UnsafeConst();
        }

        public const string NUM_PLACEHOLDER = "[%NUM%]";
        public const string URL_PLACEHOLDER = "[%URL%]";

        #region [.cctor().]
        private static CharType*         _CTM;
        private static char*             _UIM;
        private static SpecialCharType*  _SCTM;
        private static HashSet< string > _DigitsSpecEnds;
        static Tokenizer()
        {
            _UIM  = xlat_Unsafe.Inst._UPPER_INVARIANT_MAP;
            _CTM  = xlat_Unsafe.Inst._CHARTYPE_MAP;
            _SCTM = UnsafeConst.Inst._SPEC_CHARTYPE_MAP;

            _DigitsSpecEnds = new HashSet< string >( StringComparer.InvariantCultureIgnoreCase ) 
            { 
                "г", "г.", 
                "кг", "кг.",
                "см", "см.",
                "км", "км.", 
                "тыс", "тыс.",
                "млн", "млн."
            };
        }
        #endregion

        #region [.private field's.]
        private const int DEFAULT_WORDSLIST_CAPACITY = 100;
        private const int DEFAULT_WORDTOUPPERBUFFER  = 100;

        private readonly SentSplitter                 _SentSplitter;
        private readonly UrlDetector                  _UrlDetector;
        private readonly List< word_t >               _Words;
        private readonly IPosTaggerInputTypeProcessor _PosTaggerInputTypeProcessor;        
        private char*                                 _BASE;
        private char*                                 _Ptr;        
        private int                                   _StartIndex;
        private int                                   _Length;
        private ProcessSentCallbackDelegate           _OuterProcessSentCallback_Delegate;
        private char*                                 _StartPtr;
        private char*                                 _EndPtr;
        private int                                   _WordToUpperBufferSize;
        private GCHandle                              _WordToUpperBufferGCHandle;
        private char*                                 _WordToUpperBufferPtrBase;
        private SentSplitter.ProcessSentCallbackDelegate _SentSplitterProcessSentCallback_Delegate;
        private SentSplitter.ProcessSentCallbackDelegate _SentSplitterProcessSentCallback_Simple_Delegate; 
        private UmlautesNormalizer                    _UmlautesNormalizer;
        private readonly sent_t                       _NoSentsAllocateSent;
        private ProcessSentCallbackDelegate           _Dummy_ProcessSentCallbackDelegate;
        private ProcessSentCallbackDelegate           _AccumulateSents_ProcessSentCallbackDelegate;
        private readonly List< List< word_t > >       _AccumulateSents_Words;
        private bool                                  _ReplaceNumsOnPlaceholders;
        private bool                                  _IsPrevWordNumber;
        private word_t                                _LastWordNumber;
        #endregion

        #region [.ctor().]
        public Tokenizer( TokenizerConfig config, bool replaceNumsOnPlaceholder = true )
        {
            _SentSplitter = new SentSplitter( config.SentSplitterConfig );
            _Words        = new List< word_t >( DEFAULT_WORDSLIST_CAPACITY );
            _SentSplitterProcessSentCallback_Delegate        = new SentSplitter.ProcessSentCallbackDelegate( SentSplitterProcessSentCallback );
            _SentSplitterProcessSentCallback_Simple_Delegate = new SentSplitter.ProcessSentCallbackDelegate( SentSplitterProcessSentCallback_Simple );

            //--//
            ReAllocWordToUpperBuffer( DEFAULT_WORDTOUPPERBUFFER );
            _PosTaggerInputTypeProcessor = config.PosTaggerInputTypeProcessor ?? PosTaggerInputTypeProcessor_En.Inst;
            _UmlautesNormalizer          = new UmlautesNormalizer();

            _UrlDetector         = _SentSplitter.UrlDetector;
            _NoSentsAllocateSent = sent_t.CreateEmpty();
            _Dummy_ProcessSentCallbackDelegate = new ProcessSentCallbackDelegate( words => { }/*(words, urls) => { }*/ );

            _AccumulateSents_ProcessSentCallbackDelegate = new ProcessSentCallbackDelegate( Accumulate_ProcessSentCallback );
            _AccumulateSents_Words = new List< List< word_t > >( DEFAULT_WORDSLIST_CAPACITY );

            _ReplaceNumsOnPlaceholders = replaceNumsOnPlaceholder;
        }

        private void ReAllocWordToUpperBuffer( int newBufferSize )
        {
            DisposeNativeResources();

            _WordToUpperBufferSize = newBufferSize;
            var wordToUpperBuffer  = new char[ _WordToUpperBufferSize ];
            _WordToUpperBufferGCHandle = GCHandle.Alloc( wordToUpperBuffer, GCHandleType.Pinned );
            _WordToUpperBufferPtrBase  = (char*) _WordToUpperBufferGCHandle.AddrOfPinnedObject().ToPointer();
        }

        ~Tokenizer() => DisposeNativeResources();
        public void Dispose()
        {
            _SentSplitter?.Dispose();
            //_UrlDetector? .Dispose();

            DisposeNativeResources();
            GC.SuppressFinalize( this );
        }
        private void DisposeNativeResources()
        {
            if ( _WordToUpperBufferPtrBase != null )
            {
                _WordToUpperBufferGCHandle.Free();
                _WordToUpperBufferPtrBase = null;
            }
        }
        #endregion

        public IPosTaggerInputTypeProcessor InputTypeProcessor { [M(O.AggressiveInlining)] get => _PosTaggerInputTypeProcessor; }
        public UrlDetector UrlDetector { [M(O.AggressiveInlining)] get => _UrlDetector; }
        public SentSplitter SentSplitter { [M(O.AggressiveInlining)] get => _SentSplitter; }
        public bool ReplaceNumsOnPlaceholders { [M(O.AggressiveInlining)] get => _ReplaceNumsOnPlaceholders; }

        #region [.Merge urls with words.]
        /// <summary>
        /// 
        /// </summary>
        private sealed class word_by_startIndex_Comparer : IComparer< word_t >
        {
            public static word_by_startIndex_Comparer Inst { [M(O.AggressiveInlining)] get; } = new word_by_startIndex_Comparer();
            private word_by_startIndex_Comparer() { }
            public int Compare( word_t x, word_t y ) => (x.startIndex - y.startIndex);
        }

        [M(O.AggressiveInlining)] private static word_t CreateWord( url_t url )
        {
            var w = new word_t()
            {
                posTaggerInputType = PosTaggerInputType.Other,
                startIndex         = url.startIndex,
                length             = url.length,
                valueOriginal      = url.value,
                valueUpper         = url.value,
            };

            switch ( url.type )
            {
                case UrlTypeEnum.Email: w.posTaggerInputType = PosTaggerInputType.Email; /*w.posTaggerOutputType = PosTaggerOutputType.Email;*/ break;
                case UrlTypeEnum.Url  : w.posTaggerInputType = PosTaggerInputType.Url;   /*w.posTaggerOutputType = PosTaggerOutputType.Url;*/   break;
                //---default: throw (new ArgumentException( url.ToString() ));
            }

            return (w);
        }
        [M(O.AggressiveInlining)] private static void MergeUrlsToWords( List< word_t > words, List< url_t > urls )
        {
            if ( urls != null )
            {
                for ( var i = urls.Count - 1; 0 <= i; i-- )
                {
                    words.Add( CreateWord( urls[ i ] ) );
                }
                words.Sort( word_by_startIndex_Comparer.Inst );
            }
        }
        #endregion

        #region [.Run.]
        public List< word_t > Run_NoSentsNoUrlsAllocate( string text )
        {
            _OuterProcessSentCallback_Delegate = _Dummy_ProcessSentCallbackDelegate;
            fixed ( char* _base = text )
            {
                _BASE = _base;
                _NoSentsAllocateSent.Set( 0, text.Length, null );
                SentSplitterProcessSentCallback( _NoSentsAllocateSent );
            }
            _OuterProcessSentCallback_Delegate = null;

            return (_Words);
        }
        public List< word_t > Run_NoSentsAllocate( string text )
        {
            _OuterProcessSentCallback_Delegate = _Dummy_ProcessSentCallbackDelegate;
            fixed ( char* _base = text )
            {
                _BASE = _base;

                var urls = _UrlDetector.AllocateUrls( text );
                _NoSentsAllocateSent.Set( 0, text.Length, (0 < urls.Count) ? urls : null );
                SentSplitterProcessSentCallback( _NoSentsAllocateSent );
            }
            _OuterProcessSentCallback_Delegate = null;

            //---MergeUrlsToWords( _Words, _NoSentsAllocateSent.urls );
            return (_Words);
        }

        public void ___Run___v0___( string text, ProcessSentCallbackDelegate processSentCallback )
        {
            _OuterProcessSentCallback_Delegate = processSentCallback;
            fixed ( char* _base = text )
            {
                _BASE = _base;
                _SentSplitter.AllocateSents( text, _SentSplitterProcessSentCallback_Delegate );
            }
            _OuterProcessSentCallback_Delegate = null;
        }
        public void Run_SimpleSentsAllocate( string text, ProcessSentCallbackDelegate processSentCallback )
        {
            _OuterProcessSentCallback_Delegate = processSentCallback;
            fixed ( char* _base = text )
            {
                _BASE = _base;
                _SentSplitter.AllocateSents_Simple( _base, text.Length, _SentSplitterProcessSentCallback_Simple_Delegate ); //_SentSplitterProcessSentCallback_Delegate );
            }
            _OuterProcessSentCallback_Delegate = null;
        }
        
        public List< List< word_t > > ___Run___v0___( string text )
        {
            _AccumulateSents_Words.Clear();
            ___Run___v0___( text, _AccumulateSents_ProcessSentCallbackDelegate );
            return (_AccumulateSents_Words);
        }
        public List< List< word_t > > Run_SimpleSentsAllocate( string text )
        {
            _AccumulateSents_Words.Clear();
            Run_SimpleSentsAllocate( text, _AccumulateSents_ProcessSentCallbackDelegate );
            return (_AccumulateSents_Words);
        }
        private void Accumulate_ProcessSentCallback( List< word_t > words ) => _AccumulateSents_Words.Add( words.ToList( words.Count ) );
        #endregion

        [M(O.AggressiveInlining)] private void SentSplitterProcessSentCallback( sent_t sent )
        {
            _Words.Clear();
            _IsPrevWordNumber = false; _LastWordNumber = null;
            _StartIndex = sent.startIndex;
            _Length     = 0;
            _StartPtr   = _BASE + _StartIndex;
            _EndPtr     = _StartPtr + sent.length - 1;

            var urls        = sent.urls;
            var urlIndex    = 0;
            var startUrlPtr = (urls != null) ? (_BASE + urls[ 0 ].startIndex) : UnsafeConst.MAX_PTR;

            #region [.main.]
            var realyEndPtr = _EndPtr;
            _EndPtr = SkipNonLetterAndNonDigitToTheEnd();

            for ( _Ptr = _StartPtr; _Ptr <= _EndPtr; _Ptr++ )
            {
                #region [.process allocated url's.]
                if ( startUrlPtr <= _Ptr )
                {
                    #region [.code.]
                    TryCreateWordAndPut2List();

                    var lenu = urls[ urlIndex ].length;
                    #region [.skip-ignore url's.]
                    /*
                    #region [.create word. url.]
                    var lenu = urls[ urlIndex ].length;
                    var vu = new string( startUrlPtr, 0, lenu );
                    var wu = new word_t()
                    {
                        startIndex         = urls[ urlIndex ].startIndex, 
                        length             = lenu, 
                        valueOriginal      = vu,
                        valueUpper         = vu,
                        posTaggerInputType = PosTaggerInputType.Url
                    };
                    _Words.Add( wu );
                    #endregion
                    //*/
                    #endregion

                    _Ptr = startUrlPtr + lenu - 1;
                    urlIndex++;
                    startUrlPtr = (urlIndex < urls.Count) ? (_BASE + urls[ urlIndex ].startIndex) : UnsafeConst.MAX_PTR;

                    _StartIndex = (int) (_Ptr - _BASE + 1);
                    _Length     = 0;
                    continue;

                    #endregion
                }
                #endregion

                var ch = *_Ptr;
                var ct = _CTM[ ch ];
                #region [.whitespace.]
                if ( ct.IsWhiteSpace() )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion

                var pct = _SCTM[ ch ];
                #region [.dot.]
                if ( ((pct & SpecialCharType.DotChar) == SpecialCharType.DotChar) && IsUpperNextChar() )
                {
                    _Length++;
                    TryCreateWordAndPut2List();
                    continue;
                }
                #endregion

                #region [.between-letter-or-digit.]
                if ( (pct & SpecialCharType.BetweenLetterOrDigit) == SpecialCharType.BetweenLetterOrDigit )
                {
                    if ( !ct.IsHyphen() && IsBetweenLetterOrDigit() ) //always split by Hyphen-Dash
                    {
                        _Length++;
                    }
                    else
                    {
                        TryCreateWordAndPut2List();

                        #region [.merge punctuation (with white-space's).]
                        if ( !MergePunctuation( ch ) )
                            break;
                        #endregion

                        //punctuation word
                        TryCreateWordAndPut2List();
                    }

                    continue;
                }
                //с учетом того, что списки 'BetweenLetterOrDigit' и 'BetweenDigit' не пересекаются
                else if ( (pct & SpecialCharType.BetweenDigit) == SpecialCharType.BetweenDigit )
                {
                    if ( IsBetweenDigit() )
                    {
                        _Length++;
                    }
                    else
                    {
                        TryCreateWordAndPut2List();

                        #region [.merge punctuation (with white-space's).]
                        if ( !MergePunctuation( ch ) )
                            break;
                        #endregion

                        //punctuation word
                        TryCreateWordAndPut2List();
                    }

                    continue;                    
                }
                #endregion

                #region [.tokenize-different-separately.]
                if ( (pct & SpecialCharType.TokenizeDifferentSeparately) == SpecialCharType.TokenizeDifferentSeparately )
                {
                    TryCreateWordAndPut2List();

                    #region [.merge punctuation (with white-space's).]
                    if ( !MergePunctuation( ch ) )
                        break;
                    #region 
                    /*
                    _Length = 1;
                    _Ptr++;
                    for ( ; _Ptr <= _EndPtr; _Ptr++ ) 
                    {
                        var ch_next = *_Ptr;
                        if ( ch_next != ch )
                            break;

                        _Length++;
                    }
                    if ( _EndPtr < _Ptr )
                    {
                        if ( (_Length == 1) && (*_EndPtr == '\0') )
                            _Length = 0;
                        break;
                    }
                    _Ptr--;
                    */
                    #endregion
                    #endregion

                    //punctuation word
                    TryCreateWordAndPut2List();

                    continue;
                }
                #endregion

                #region [.interprete-as-whitespace.]
                if ( (pct & SpecialCharType.InterpreteAsWhitespace) == SpecialCharType.InterpreteAsWhitespace )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion

                #region [.increment length.]
                _Length++;
                #endregion
            }
            #endregion

            #region [.last word.]
            TryCreateWordAndPut2List();
            #endregion

            #region [.tail punctuation.]
            for ( _EndPtr = realyEndPtr; _Ptr <= _EndPtr; _Ptr++ )
            {
                var ch = *_Ptr;
                var ct = *(_CTM + ch);
                #region [.whitespace.]
                if ( ct.IsWhiteSpace() )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion
                
                var nct = _SCTM[ ch ];
                #region [.tokenize-different-separately.]
                if ( (nct & SpecialCharType.TokenizeDifferentSeparately) == SpecialCharType.TokenizeDifferentSeparately )
                {
                    TryCreateWordAndPut2List();

                    #region [.merge punctuation (with white-space's).]
                    if ( !MergePunctuation( ch ) )
                        break;
                    #endregion

                    //punctuation word
                    TryCreateWordAndPut2List();

                    continue;
                }
                #endregion

                #region [.interprete-as-whitespace.]
                if ( (nct & SpecialCharType.InterpreteAsWhitespace) == SpecialCharType.InterpreteAsWhitespace )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion

                #region [.increment length.]
                _Length++;
                #endregion
            }
            #endregion

            #region [.last punctuation.]
            TryCreateWordAndPut2List();
            #endregion

            MergeUrlsToWords( _Words, sent.urls );
            _OuterProcessSentCallback_Delegate( _Words/*, sent.urls*/ );
        }
        /// <summary>
        /// always unstick ["." (dot('s))] from end of tokens
        /// </summary>
        [M(O.AggressiveInlining)] private void SentSplitterProcessSentCallback_Simple( sent_t sent )
        {
            _Words.Clear();
            _IsPrevWordNumber = false; _LastWordNumber = null;
            _StartIndex = sent.startIndex;
            _Length     = 0;
            _StartPtr   = _BASE + _StartIndex;
            _EndPtr     = _StartPtr + sent.length - 1;

            var urls        = sent.urls;
            var urlIndex    = 0;
            var startUrlPtr = (urls != null) ? (_BASE + urls[ 0 ].startIndex) : UnsafeConst.MAX_PTR;

            #region [.main.]
            var realyEndPtr = _EndPtr;
            _EndPtr = SkipNonLetterAndNonDigitToTheEnd();

            for ( _Ptr = _StartPtr; _Ptr <= _EndPtr; _Ptr++ )
            {
                #region [.process allocated url's.]
                if ( startUrlPtr <= _Ptr )
                {
                    #region [.code.]
                    TryCreateWordAndPut2List();

                    var lenu = urls[ urlIndex ].length;
                    #region [.skip-ignore url's.]
                    /*
                    #region [.create word. url.]
                    var lenu = urls[ urlIndex ].length;
                    var vu = new string( startUrlPtr, 0, lenu );
                    var wu = new word_t()
                    {
                        startIndex         = urls[ urlIndex ].startIndex, 
                        length             = lenu, 
                        valueOriginal      = vu,
                        valueUpper         = vu,
                        posTaggerInputType = PosTaggerInputType.Url
                    };
                    _Words.Add( wu );
                    #endregion
                    //*/
                    #endregion

                    _Ptr = startUrlPtr + lenu - 1;
                    urlIndex++;
                    startUrlPtr = (urlIndex < urls.Count) ? (_BASE + urls[ urlIndex ].startIndex) : UnsafeConst.MAX_PTR;

                    _StartIndex = (int) (_Ptr - _BASE + 1);
                    _Length     = 0;
                    continue;

                    #endregion
                }
                #endregion

                var ch = *_Ptr;
                var ct = _CTM[ ch ];
                #region [.whitespace.]
                if ( ct.IsWhiteSpace() )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion

                var pct = _SCTM[ ch ];
                #region [.dot.]
                if ( ((pct & SpecialCharType.DotChar) == SpecialCharType.DotChar) && !IsDigitNextChar() )
                {
                    TryCreateWordAndPut2List();

                    #region [.merge punctuation (starts with dot) (with white-space's).]
                    if ( !MergePunctuation( ch ) )
                        break;
                    #endregion

                    //punctuation (dot) word
                    TryCreateWordAndPut2List();
                    continue;
                }
                #endregion

                #region [.between-letter-or-digit.]
                if ( (pct & SpecialCharType.BetweenLetterOrDigit) == SpecialCharType.BetweenLetterOrDigit )
                {
                    if ( !ct.IsHyphen() && IsBetweenLetterOrDigit() ) //always split by Hyphen-Dash
                    {
                        _Length++;
                    }
                    else
                    {
                        TryCreateWordAndPut2List();

                        #region [.merge punctuation (with white-space's).]
                        if ( !MergePunctuation( ch ) )
                            break;
                        #endregion

                        //punctuation word
                        TryCreateWordAndPut2List();
                    }

                    continue;
                }
                //с учетом того, что списки 'BetweenLetterOrDigit' и 'BetweenDigit' не пересекаются
                else if ( (pct & SpecialCharType.BetweenDigit) == SpecialCharType.BetweenDigit )
                {
                    if ( IsBetweenDigit() )
                    {
                        _Length++;
                    }
                    else
                    {
                        TryCreateWordAndPut2List();

                        #region [.merge punctuation (with white-space's).]
                        if ( !MergePunctuation( ch ) )
                            break;
                        #endregion

                        //punctuation word
                        TryCreateWordAndPut2List();
                    }

                    continue;                    
                }
                #endregion

                #region [.tokenize-different-separately.]
                if ( (pct & SpecialCharType.TokenizeDifferentSeparately) == SpecialCharType.TokenizeDifferentSeparately )
                {
                    TryCreateWordAndPut2List();

                    #region [.merge punctuation (with white-space's).]
                    if ( !MergePunctuation( ch ) )
                        break;
                    #endregion

                    //punctuation word
                    TryCreateWordAndPut2List();

                    continue;
                }
                #endregion

                #region [.interprete-as-whitespace.]
                if ( (pct & SpecialCharType.InterpreteAsWhitespace) == SpecialCharType.InterpreteAsWhitespace )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion

                #region [.increment length.]
                _Length++;
                #endregion
            }
            #endregion

            #region [.last word.]
            TryCreateWordAndPut2List();
            #endregion

            #region [.tail punctuation.]
            for ( _EndPtr = realyEndPtr; _Ptr <= _EndPtr; _Ptr++ )
            {
                var ch = *_Ptr;
                var ct = *(_CTM + ch);
                #region [.whitespace.]
                if ( ct.IsWhiteSpace() )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion
                
                var nct = _SCTM[ ch ];
                #region [.tokenize-different-separately.]
                if ( (nct & SpecialCharType.TokenizeDifferentSeparately) == SpecialCharType.TokenizeDifferentSeparately )
                {
                    TryCreateWordAndPut2List();

                    #region [.merge punctuation (with white-space's).]
                    if ( !MergePunctuation( ch ) )
                        break;
                    #endregion

                    //punctuation word
                    TryCreateWordAndPut2List();

                    continue;
                }
                #endregion

                #region [.interprete-as-whitespace.]
                if ( (nct & SpecialCharType.InterpreteAsWhitespace) == SpecialCharType.InterpreteAsWhitespace )
                {
                    TryCreateWordAndPut2List();

                    _StartIndex++;
                    continue;
                }
                #endregion

                #region [.increment length.]
                _Length++;
                #endregion
            }
            #endregion

            #region [.last punctuation.]
            TryCreateWordAndPut2List();
            #endregion

            MergeUrlsToWords( _Words, sent.urls );
            _OuterProcessSentCallback_Delegate( _Words/*, sent.urls*/ );
        }

        [M(O.AggressiveInlining)] private word_t Create_NUM_PLACEHOLDER_Word()
            => new word_t()
            {
                startIndex         = _StartIndex, 
                length             = _Length, 
                valueOriginal      = NUM_PLACEHOLDER,
                valueUpper         = NUM_PLACEHOLDER,
                posTaggerInputType = PosTaggerInputType.Num,
                extraWordType      = ExtraWordType.IntegerNumber
            };
        private void TryCreateWordAndPut2List()
        {
            if ( _Length != 0 )
            {
                var startPtr = _BASE + _StartIndex;

                #region [.replace or skip second-and-next-num-words.]
                if ( _ReplaceNumsOnPlaceholders )
                {
                    var is_num = IsDigits_WithPunctuations_WithSpecEnds( startPtr, _Length );//---IsDigitsWithPunctuations( startPtr, _Length );
                    if ( is_num )
                    {
                        if ( !_IsPrevWordNumber )
                        {
                            _IsPrevWordNumber = true;
                            _LastWordNumber   = Create_NUM_PLACEHOLDER_Word();
                            _Words.Add( _LastWordNumber );
                        }
                        #region comm. [.skip num-word. increment start-index.]
                        //_StartIndex += _Length;
                        //_Length      = 0;
                        //return;
                        #endregion                    
                        goto EXIT;
                    }
                    else if ( _LastWordNumber != null )
                    {
                        var i = 1;
                        for ( var ln = _StartIndex - _LastWordNumber.startIndex; i < ln; i++ )
                        {
                            if ( !_CTM[ *(startPtr - i) ].IsWhiteSpace() )
                            {
                                break;
                            }
                        }
                        _LastWordNumber.length = _StartIndex - _LastWordNumber.startIndex - i + 1;
                        _LastWordNumber = null;
                    }
                    _IsPrevWordNumber = false;
                }
                #endregion

                #region [.to upper invariant & pos-tagger-list & etc.]
                if ( _WordToUpperBufferSize < _Length )
                {
                    ReAllocWordToUpperBuffer( _Length );
                }                
                for ( int i = 0; i < _Length; i++ )
                {
                    *(_WordToUpperBufferPtrBase + i) = *(_UIM + *(startPtr + i));
                }
                var valueUpper = new string( _WordToUpperBufferPtrBase, 0, _Length );
                #endregion

                #region [.create word.]
                var valueOriginal = new string( _BASE, _StartIndex, _Length );
                var word = new word_t()
                {
                    startIndex    = _StartIndex, 
                    length        = _Length, 
                    valueOriginal = valueOriginal,
                    valueUpper    = valueUpper,
                };
                #endregion

                #region [.posTaggerInputType.]
                ( word.posTaggerInputType, word.extraWordType) = _PosTaggerInputTypeProcessor.GetPosTaggerInputType( _BASE + _StartIndex, _Length );

                if ( ((word.extraWordType & ExtraWordType.HasUmlautes) == ExtraWordType.HasUmlautes) /*&& (_UmlautesNormalizer != null) //---ALWAYE NOT_NULL(?)---// */ )
                {
                    //---word.valueOriginal__UmlautesNormalized = _UmlautesNormalizer.Normalize( _BASE + _StartIndex, _Length );
                    word.valueUpper__UmlautesNormalized = _UmlautesNormalizer.Normalize_ToUpper( _WordToUpperBufferPtrBase, _Length );
                }
                #endregion

                #region [.put-2-list.]
                Clear_valueOriginal( word );

                _Words.Add( word );
                #endregion
            EXIT:
                #region [.inctement start-index.]
                _StartIndex += _Length;
                _Length      = 0;
                #endregion
            }
        }

        [M(O.AggressiveInlining)] private char* SkipNonLetterAndNonDigitToTheEnd()
        {
            for ( char* ptr = _EndPtr; _StartPtr <= ptr; ptr-- )
            {
                var ct = *(_CTM + *ptr);
                if ( ct.IsLetter() || ct.IsDigit() )
                {
                    #region [.если на конце предложения одиночная буква большая, то точку не отрывать.]
                    if ( ct.IsUpper() )
                    {
                        var p = ptr - 1;
                        if ( (_StartPtr == p) || ((_StartPtr < p) && (*(_CTM + *p)).IsWhiteSpace()) )
                        {
                            p = ptr + 1;
                            if ( (p == _EndPtr) || ((p < _EndPtr) && (*(_CTM + *(p + 1))).IsWhiteSpace()) )
                            {
                                if ( xlat.IsDot( *p ) )
                                return (p);
                            }
                        }
                    }
                    #endregion

                    return (ptr);
                }
            }
            return (_StartPtr - 1);
        }

        [M(O.AggressiveInlining)] private bool IsBetweenLetterOrDigit()
        {
            if ( _Ptr <= _StartPtr )
                return (false);

            var ch = *(_Ptr - 1);
            var ct = *(_CTM + ch);
            if ( !ct.IsLetter() && !ct.IsDigit() )
            {
                return (false);
            }

            var p = _Ptr + 1;
            if ( _EndPtr <= p )
            {
                if ( _EndPtr < p )
                    return (false);
                ch = *p;
                if ( ch == '\0' )
                    return (false);
            }
            else
            {
                ch = *p;
            }
            ct = *(_CTM + ch);
            if ( !ct.IsLetter() && !ct.IsDigit() )
            {
                return (false);
            }

            return (true);
        }
        [M(O.AggressiveInlining)] private bool IsBetweenDigit()
        {
            if ( _Ptr <= _StartPtr )
                return (false);

            var ch = *(_Ptr - 1);
            var ct = *(_CTM + ch);
            if ( !ct.IsDigit() )
            {
                return (false);
            }

            var p = _Ptr + 1;
            if ( _EndPtr <= p )
            {
                if ( _EndPtr < p )
                    return (false);
                ch = *p;
                if ( ch == '\0' )
                    return (false);
            }
            else
            {
                ch = *p;
            }
            ct = *(_CTM + ch);
            if ( !ct.IsDigit() )
            {
                return (false);
            }

            return (true);
        }
        [M(O.AggressiveInlining)] private bool IsUpperNextChar()
        {
            var p = _Ptr + 1;
            var ch = default(char);
            if ( _EndPtr <= p )
            {
                if ( _EndPtr < p )
                    return (false);
                ch = *p;
                if ( ch == '\0' )
                    return (false);
            }
            else
            {
                ch = *p;
            }

            var ct = *(_CTM + ch);
            if ( !ct.IsUpper() )
            {
                return (false);
            }

            return (true);
        }
        [M(O.AggressiveInlining)] private bool IsDigitNextChar()
        {
            var p = _Ptr + 1;
            var ch = default(char);
            if ( _EndPtr <= p )
            {
                if ( _EndPtr < p )
                    return (false);
                ch = *p;
                if ( ch == '\0' )
                    return (false);
            }
            else
            {
                ch = *p;
            }

            var ct = *(_CTM + ch);
            if ( !ct.IsDigit() )
            {
                return (false);
            }

            return (true);
        }

        [M(O.AggressiveInlining)] private bool MergePunctuation( char begining_ch )
        {
            _Length = 1;
            _Ptr++;
            var whitespace_length = 0;
            for ( ; _Ptr <= _EndPtr; _Ptr++ ) 
            {                
                var ch_next = *_Ptr;
                var ct = *(_CTM  + ch_next);
                if ( ct.IsWhiteSpace() )
                {
                    whitespace_length++;
                    continue;
                }

                var nct = *(_SCTM + ch_next);
                if ( (nct & SpecialCharType.InterpreteAsWhitespace) == SpecialCharType.InterpreteAsWhitespace )
                {
                    whitespace_length++;
                    continue;
                }

                if ( ch_next == begining_ch )
                {
                    _Length += whitespace_length + 1;
                    whitespace_length = 0;
                    continue;
                }

                break;
            }
            if ( _EndPtr < _Ptr )
            {
                if ( (_Length == 1) && (*_EndPtr == '\0') )
                    _Length = 0;
                return (false);
            }
            _Ptr -= whitespace_length + 1;

            return (true);
        }

        [M(O.AggressiveInlining)] public string NormalizeUmlautes( string word ) => _UmlautesNormalizer.Normalize( word );
        [M(O.AggressiveInlining)] public string NormalizeUmlautes_ToUpper( string word ) => _UmlautesNormalizer.Normalize_ToUpper( word );

        [M(O.AggressiveInlining)] unsafe private static bool IsDigitsWithPunctuations( char* ptr, int length )
        {
            var hasDigits = false;
            for ( var i = length - 1; 0 <= i; i-- )
            {
                var ct = _CTM[ ptr[ i ] ];
                hasDigits |= ct.IsDigit();
                if ( !hasDigits && !ct.IsPunctuation() )
                {
                    return (false);
                }
            }
            return (hasDigits);
        }


        private StringBuilder _Buf = new StringBuilder( 0x10 );
        [M(O.AggressiveInlining)] unsafe private bool IsDigits_WithPunctuations_WithSpecEnds( char* ptr, int length )
        {
            var hasDigits = false;
            for ( var i = 0; i < length; i++ )
            {
                var ct = _CTM[ ptr[ i ] ];
                hasDigits |= ct.IsDigit();
                if ( /*!hasDigits*/!ct.IsDigit() && !ct.IsPunctuation() )
                {
                    if ( hasDigits )
                    {
                        var end = _Buf.Append( ptr + i, length - i ).ToString(); _Buf.Clear();
                        return (_DigitsSpecEnds.Contains( end ));
                    }
                    return (false);
                }
            }
            return (hasDigits);
        }

        [M(O.AggressiveInlining)] private static void Clear_valueOriginal( word_t w )
        {            
            if ( (w.extraWordType & ExtraWordType.Punctuation) == ExtraWordType.Punctuation )
            {
                var v = w.valueOriginal;
                if ( w.length == 1 )
                {
                    var ch = w.valueOriginal[ 0 ];
                    switch ( ch )
                    {
                        case ':': case '.': case ',': case ';': case '?': case '!': case '(': case ')': case '/': case '%': case '&': case '…': break;
                        default:
                            var ct = xlat.CHARTYPE_MAP[ ch ];
                            if ( xlat.IsHyphen( ct ) )
                            {
                                if ( ch != '-' )
                                {
                                    v = "-";
                                }
                            }
                            else if ( (ct & CharType.IsQuote) == CharType.IsQuote )
                            {
                                switch ( ch )
                                {
                                    case '\"': case '\'': case '[': case ']': break;
                                    default:
                                        v = "\"";
                                        break;
                                }
                            }
#if DEBUG && XXX
                            else
                            {
                                switch ( ch )
                                {
                                    case '[': case ']': case '{': case '}': break;
                                    default:
                                        int ttttt = 0;
                                        break;
                                }
                            } 
#endif
                            break;
                    }
                }
                else if ( v == "''" )
                {
                    v = "\"";
                }
                else if ( v == ",," )
                {
                    v = ",";
                }
                else if ( v == "--" )
                {
                    v = "-";
                }
#if DEBUG && XXX
                else if ( v != "..." && v != "//" && v != ".+" )
                {
                    int tttttt = 0; 
                }
#endif
                w.valueOriginal = v;
            }
        }

        
        private static char[] _LOWER_INVARIANT_MAP = xlat.Create_LOWER_INVARIANT_MAP();
        [M(O.AggressiveInlining)] unsafe public static string ToPosTaggerToken( word_t w, int maxEndingLength )
        {
            switch ( w.posTaggerInputType )
            {
                case PosTaggerInputType.Num: return (NUM_PLACEHOLDER);
                case PosTaggerInputType.Url:
                case PosTaggerInputType.Email: return (URL_PLACEHOLDER);
                default:
                    return (ToPosTaggerToken( w.valueOriginal, maxEndingLength ));
            }
        }
        [M(O.AggressiveInlining)] unsafe public static string ToPosTaggerToken( string token, int maxEndingLength )
        {
            if ( StringsHelper.IsEqual( NUM_PLACEHOLDER, token ) )
            {
                return (NUM_PLACEHOLDER); //(token);
            }

            Debug.Assert( token.Length < 0x1000 );

            var len       = token.Length;
            var new_len   = 0;
            var new_chars = stackalloc char[ len ];
            fixed ( char* p = token )
            {
                for ( var i = 0; i < len; i++ )
                {
                    var ch = _LOWER_INVARIANT_MAP[ p[ i ] ];
                    if ( ch == 'ё' ) ch = 'е';
                    if ( char.GetUnicodeCategory( ch ) != UnicodeCategory.NonSpacingMark )
                    {
                        new_chars[ new_len++ ] = ch;
                    }
                }
            }
            var d = new_len - maxEndingLength - 1;
            if ( 0 <= d )
            {
                new_chars[ d ] = '_';
                return (new string( new_chars, d, maxEndingLength + 1 ));
            }
            return (new string( new_chars, 0, new_len ));

            #region comm.
            //var new_token = new string( token.ToLowerInvariant().Replace( 'ё', 'е' ).Where( ch => char.GetUnicodeCategory( ch ) != UnicodeCategory.NonSpacingMark ).ToArray() );
            //var d = new_token.Length - maxEndingLength;
            //if ( 0 < d )
            //{
            //    new_token = '_' + new_token.Substring( d, maxEndingLength );
            //}
            //return (new_token); 
            #endregion
        }
        /*private static string ToPosTaggerToken__v0( string token, int maxEndingLength )
        {
            var new_token = new string( token.ToLowerInvariant().Replace( 'ё', 'е' ).Where( ch => char.GetUnicodeCategory( ch ) != UnicodeCategory.NonSpacingMark ).ToArray() );
            var d = new_token.Length - maxEndingLength;
            if ( 0 < d )
            {
                new_token = '_' + new_token.Substring( d, maxEndingLength );
            }
            return (new_token);
        }
        //*/

        [M(O.AggressiveInlining)] public static string GetOriginalValue( word_t w, string originalText )
        {
            if ( w.posTaggerInputType == PosTaggerInputType.Num )
            {
                return (originalText.Substring( w.startIndex, w.length ));
            }
            return (w.valueOriginal);
        }
    }
}
