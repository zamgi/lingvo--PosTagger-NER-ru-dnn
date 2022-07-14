using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Xml.Linq;

using Lingvo.PosTagger.Tokenizing;
using Lingvo.PosTagger.Urls;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.SentSplitting
{
    /// <summary>
    /// 
    /// </summary>
    internal struct before_no_proper_t
    {
        public bool UnstickFromDigits;
        public bool DigitsAfter;
        public before_no_proper_t( bool unstickFromDigits, bool digitsAfter ) => (UnstickFromDigits, DigitsAfter) = (unstickFromDigits, digitsAfter);
#if DEBUG
        public override string ToString() => (UnstickFromDigits ? "unstick-from-digits: true" : null) +
                                             (DigitsAfter       ? " digits-after: true"       : null);
#endif
    }
    /// <summary>
    /// 
    /// </summary>
    internal struct before_proper_or_number_t
    {
        public before_proper_or_number_t( bool digitsBefore, bool slashBefore, bool unstickFromDigits )
        {
            DigitsBefore      = digitsBefore;
            SlashBefore       = slashBefore;
            UnstickFromDigits = unstickFromDigits;

            DigitsBeforeOrSlashBefore = DigitsBefore | SlashBefore;
        }

        public bool DigitsBefore;
        public bool SlashBefore;
        public bool UnstickFromDigits;

        public bool DigitsBeforeOrSlashBefore { [M(O.AggressiveInlining)] get; }
#if DEBUG
        public override string ToString()
        {
            var v = default( string );
            if ( DigitsBefore )
            {
                v = "digits-before: " + DigitsBefore;
            }
            if ( SlashBefore )
            {
                if ( v != null ) v += ", ";
                v += "slash-before: " + SlashBefore;
            }
            if ( UnstickFromDigits )
            {
                if ( v != null ) v += ", ";
                v += "unstick-from-digits: " + UnstickFromDigits;
            }
            return (v);
        } 
#endif
    }

    /// <summary>
    /// 
    /// </summary>
    unsafe public class SentSplitterModel : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        internal readonly struct hashset_t
        {
            public hashset_t( Set< string > values )
            {
                Values          = values;
                ValuesMaxLength = values.GetItemMaxLength();
                ValuesMinLength = values.GetItemMinLength();
            }

            public Set< string > Values          { [M(O.AggressiveInlining)] get; }
            public int           ValuesMaxLength { [M(O.AggressiveInlining)] get; }
            public int           ValuesMinLength { [M(O.AggressiveInlining)] get; }
        }

        /// <summary>
        /// 
        /// </summary>
        [Flags] internal enum SentCharType : byte
        {
            __UNDEFINE__ = 0x0,

            Unconditional                        = 0x1,
            SmileBegin                           = 1 << 1,
            ExcludeInBracketAndQuote             = 1 << 2,
            Dot                                  = 1 << 3,
            ThreeDot                             = 1 << 4,
            RomanDigit                           = 1 << 5,
            AfterThreeDotAllowedPunctuation      = 1 << 6,
            AfterBracketAllowedPunctuation4QMEP  = 1 << 7, 
            //Colon                                = 1 << 6,
            //BeforeProperOrNumberDigitsBeforeChar = 1 << 7,
        }

        public SentSplitterModel( string sentSplitterResourcesXmlFilename ) => Init( XDocument.Load( sentSplitterResourcesXmlFilename ) );
        public SentSplitterModel( StreamReader sentSplitterResourcesXmlStreamReader ) => Init( XDocument.Load( sentSplitterResourcesXmlStreamReader ) );
        private void Init( XDocument xdoc )
        {
            //-file-extensions-
            var fileExtensions = from xe in xdoc.Root.Element( "file-extensions" ).Elements()
                                 select 
                                    xe.Value.Trim().TrimStartDot();
            _FileExtensions = new hashset_t( fileExtensions.ToSet( toUpperInvariant: true ) );

            //-streets-ends-(as-end-of-complex-words)-
            var streetsEndsAsEndComplexWord = (from xe in xdoc.Root.Element( "streets-ends-as-end-of-complex-words" ).Elements()
                                               select xe.Value.Trim()
                                              );//.ToArray();
            _StreetsEndsAsEndComplexWord = new streets_ends_as_end_complex_word_t( streetsEndsAsEndComplexWord );

            //-streets-ends-(as-separate-words)-
            var streetsEndsAsSeparateWord = (from xe in xdoc.Root.Element( "streets-ends-as-separate-words" ).Elements()
                                             select
                                               xe.ToBeforeNoProper_ngrams()
                                            )
                                            .ToArray();
            StreetsEndsAsSeparateWordSearcher = new Searcher< before_no_proper_t >( streetsEndsAsSeparateWord );

            //-before-no-proper-
            var beforeNoProper = (from xe in xdoc.Root.Element( "before-no-proper" ).Elements()
                                  select 
                                     xe.ToBeforeNoProper_ngrams()
                                 )
                                 .ToArray();
            BeforeNoProperSearcher = new Searcher< before_no_proper_t >( beforeNoProper );

            //-before-proper-or-number-
            var beforeProperOrNumber = (from xe in xdoc.Root.Element( "before-proper-or-number" ).Elements()
                                        select 
                                           xe.ToBeforeProperOrNumber_ngrams()
                                       ).ToArray();
            BeforeProperOrNumberSearcher = new Searcher< before_proper_or_number_t >( beforeProperOrNumber );

            //--//
            var SENTCHARTYPE_MAP = InitializeSentPotentialEnds( beforeNoProper, beforeProperOrNumber );

            _SENTCHARTYPE_MAP_GCHandle = GCHandle.Alloc( SENTCHARTYPE_MAP, GCHandleType.Pinned );
            _SENTCHARTYPE_MAP          = (SentCharType*) _SENTCHARTYPE_MAP_GCHandle.AddrOfPinnedObject().ToPointer();
        }

        ~SentSplitterModel() => DisposeNativeResources();
        public void Dispose()
        {
            DisposeNativeResources();
            GC.SuppressFinalize( this );
        }
        private void DisposeNativeResources()
        {
            if ( _SENTCHARTYPE_MAP != null )
            {
                _SENTCHARTYPE_MAP_GCHandle.Free();
                _SENTCHARTYPE_MAP = null;
            }
        }

        private hashset_t _FileExtensions;
        private streets_ends_as_end_complex_word_t _StreetsEndsAsEndComplexWord;

        internal ref hashset_t                          FileExtensions               { [M(O.AggressiveInlining)] get => ref _FileExtensions; }
        internal Searcher< before_no_proper_t >         BeforeNoProperSearcher       { [M(O.AggressiveInlining)] get; private set; }
        internal Searcher< before_proper_or_number_t >  BeforeProperOrNumberSearcher { [M(O.AggressiveInlining)] get; private set; }
        internal ref streets_ends_as_end_complex_word_t StreetsEndsAsEndComplexWord  { [M(O.AggressiveInlining)] get => ref _StreetsEndsAsEndComplexWord; }
        internal Searcher< before_no_proper_t >         StreetsEndsAsSeparateWordSearcher { [M(O.AggressiveInlining)] get; private set; }

        internal Set< string > UnstickFromDigits { [M(O.AggressiveInlining)] get; private set; }
        internal int GetValuesMaxLength()
        {
            var valuesMaxLengths = new[]
            {
                _FileExtensions             .ValuesMaxLength,
                _StreetsEndsAsEndComplexWord.ValuesMinLength,
            };
            return (valuesMaxLengths.Max());
        }
        internal int GetNgramMaxLength() => Math.Max( BeforeNoProperSearcher.NgramMaxLength, BeforeProperOrNumberSearcher.NgramMaxLength );

        private GCHandle _SENTCHARTYPE_MAP_GCHandle;
        internal SentCharType* _SENTCHARTYPE_MAP { [M(O.AggressiveInlining)] get; private set; }

        private byte[] InitializeSentPotentialEnds( ngram_t< before_no_proper_t >[] beforeNoProper, ngram_t< before_proper_or_number_t >[] beforeProperOrNumber )
        {
            var SENTCHARTYPE_MAP = new byte[ char.MaxValue + 1 ];

            SENTCHARTYPE_MAP[ '!' ] |= (byte) SentCharType.ExcludeInBracketAndQuote;
            SENTCHARTYPE_MAP[ '?' ] |= (byte) SentCharType.ExcludeInBracketAndQuote;
            SENTCHARTYPE_MAP[ '…' ] |= (byte) SentCharType.ExcludeInBracketAndQuote | (byte) SentCharType.ThreeDot;
               
            //-un-conditional-
            SENTCHARTYPE_MAP[ '\n' ] = (byte) SentCharType.Unconditional;

            //-dot-
            SENTCHARTYPE_MAP[ '.' ] = (byte) SentCharType.Dot;

            //-colon-
            //---SENTCHARTYPE_MAP[ ':' ] |= SentCharType.Colon;

            //-after ThreeDot allowed punctuation-
            SENTCHARTYPE_MAP[ ';' ] |= (byte) SentCharType.AfterThreeDotAllowedPunctuation;
            SENTCHARTYPE_MAP[ ':' ] |= (byte) SentCharType.AfterThreeDotAllowedPunctuation | (byte) SentCharType.AfterBracketAllowedPunctuation4QMEP;
            SENTCHARTYPE_MAP[ ',' ] |= (byte) SentCharType.AfterThreeDotAllowedPunctuation | (byte) SentCharType.AfterBracketAllowedPunctuation4QMEP;
            for ( var c = char.MinValue; /*c <= char.MaxValue*/; c++ )
            {
                var ct = xlat.CHARTYPE_MAP[ c ];
                if ( (ct & CharType.IsHyphen) == CharType.IsHyphen )  //if ( xlat.IsHyphen( c ) )
                {
                    SENTCHARTYPE_MAP[ c ] |= (byte) SentCharType.AfterThreeDotAllowedPunctuation | (byte) SentCharType.AfterBracketAllowedPunctuation4QMEP;
                }
                else 
                if ( (ct & CharType.IsQuote) == CharType.IsQuote )
                {
                    SENTCHARTYPE_MAP[ c ] |= (byte) SentCharType.AfterThreeDotAllowedPunctuation;
                }

                if ( c == char.MaxValue )
                {
                    break;
                }
            }

            //roman-digit
            SENTCHARTYPE_MAP[ 'I' ] |= (byte) SentCharType.RomanDigit;
            SENTCHARTYPE_MAP[ 'V' ] |= (byte) SentCharType.RomanDigit;
            SENTCHARTYPE_MAP[ 'X' ] |= (byte) SentCharType.RomanDigit;
            SENTCHARTYPE_MAP[ 'C' ] |= (byte) SentCharType.RomanDigit;
            SENTCHARTYPE_MAP[ 'L' ] |= (byte) SentCharType.RomanDigit;
            SENTCHARTYPE_MAP[ 'M' ] |= (byte) SentCharType.RomanDigit;            

            foreach ( var ngram in beforeProperOrNumber )
            {
                if ( ngram.value.DigitsBefore )
                {
                    if ( ngram.words.Length      != 1 || 
                         ngram.words[ 0 ].Length != 2 ||
                         ngram.words[ 0 ][ 1 ]   != '.'
                       )
                    {
                        throw (new ArgumentException("Value for <before-proper-or-number> items with attribute [ @digits-before='true' ] must be single word length of 2 with dot on end, wrong value: " + ngram));
                    }
                }
            }


            UnstickFromDigits = new Set< string >();
            foreach ( var ngram in beforeNoProper )
            {
                if ( ngram.value.UnstickFromDigits )
                {
                    UnstickFromDigits.Add( ngram.words[ 0 ] );
                }
            }
            foreach ( var ngram in beforeProperOrNumber )
            {
                if ( ngram.value.UnstickFromDigits )
                {
                    UnstickFromDigits.Add( ngram.words[ 0 ] );
                }
            }

            return (SENTCHARTYPE_MAP);
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public sealed class SentSplitterConfig : IDisposable
    {
        public SentSplitterConfig( string sentSplitterResourcesXmlFilename, string urlDetectorResourcesXmlFilename )
        {
            Model             = new SentSplitterModel( sentSplitterResourcesXmlFilename );
            UrlDetectorConfig = new UrlDetectorConfig( urlDetectorResourcesXmlFilename  );
        }
        public SentSplitterConfig( StreamReader sentSplitterResourcesXmlStreamReader, StreamReader urlDetectorResourcesXmlStreamReader )
        {
            Model             = new SentSplitterModel( sentSplitterResourcesXmlStreamReader );
            UrlDetectorConfig = new UrlDetectorConfig( urlDetectorResourcesXmlStreamReader  );
        }
        public void Dispose() => Model.Dispose();

        public SentSplitterModel Model             { get; }
        public UrlDetectorConfig UrlDetectorConfig { get; set; }
    }
}
