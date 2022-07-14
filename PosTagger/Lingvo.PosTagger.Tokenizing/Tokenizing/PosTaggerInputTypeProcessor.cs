using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    public interface IPosTaggerInputTypeProcessor
    {
        unsafe (PosTaggerInputType posTaggerInputType, ExtraWordType extraWordType) GetPosTaggerInputType( char* _base, int length );
        (PosTaggerInputType posTaggerInputType, ExtraWordType extraWordType) GetPosTaggerInputType( word_t word );
    }

    /// <summary>
    /// Обработчик Графематических характеристик. Прикрепляет к словам определенные признаки
    /// </summary>
    unsafe public sealed class PosTaggerInputTypeProcessor_En : IPosTaggerInputTypeProcessor
    {
        #region [.description.]
        /*
        (allC)  - все заглавные буквы на латинице (больше одной), нет точек, не содержит цифр, между буквами может быть тире : NATO;
        (allCP) - все заглавные буквы (больше одной) с точкой (точками), без тире: U.N.;
        (mixC)  - смешенные буквы: заглавные и строчные, первая буква - заглавная, между буквами может быть тире, точка: St.-Petersburg , FireFox, Google.Maps
        (mixCP) - две и более заглавных подряд  с точкой (точками). Между буквами может быть тире: V.IVI.PA;
        (latC)  - хотя бы одна римская цифра буква (без точки), не содержит арабских цифр, между буквами может быть тире: XXI;
        (oneC)  - одна заглавная буква без точки и не цифра: F;
        (oneCP) - одна заглавная буква с точкой, перед буквой может быть тире: F. ;
        (Z)     - только первая заглавная: Thatcher;
        (numC)  - начинается с заглавной буквы и содержит хотя бы одну цифру, может содержать строчные, между буквами может быть тире: G8;
        (iProd) - первые строчные латиница без точки + заглавная, между буквами может быть тире: iPod.
        (Q)     - кавычки ["«“”»]
        (NUM)   - цифры в любой комбинации со знаками препинаний без букв: ["2,4", "10000", "2.456.542", "8:45"]
        (O)     - other's
        */
        #endregion

        private static CharType* _CTM;
        static PosTaggerInputTypeProcessor_En() => _CTM = xlat_Unsafe.Inst._CHARTYPE_MAP;

        public static PosTaggerInputTypeProcessor_En Inst { get; } = new PosTaggerInputTypeProcessor_En();
        private PosTaggerInputTypeProcessor_En() { }

        #region comm.
        /*/// <summary>
        /// Слово на латыни?
        /// </summary>
        [M(O.AggressiveInlining)] unsafe private static bool IsLatin( char* _base, int length )
        {
            var hasLatinLetter = false;
            for ( int i = 0; i < length; i++ )
            {
                var ch = *(_base + i);

                if ( ('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z') )
                {
                    hasLatinLetter = true;
                    continue;
                }

                if ( (*(_CTM + ch) & CharType.IsLetter) == CharType.IsLetter )
                {
                    return (false);
                }
            }

            return (hasLatinLetter);
        }//*/
        #endregion

        /// <summary>
        /// Римская цифра?
        /// </summary>
        [M(O.AggressiveInlining)] private static bool IsRomanSymbol( char ch )
		{
            switch ( ch )
            {
                case 'I':
                case 'V':
                case 'X':
                case 'L':
                case 'C':
                case 'D':
                case 'M':
                    return (true);
            }
			return (false);
		}

        [M(O.AggressiveInlining)] unsafe public (PosTaggerInputType posTaggerInputType, ExtraWordType extraWordType) GetPosTaggerInputType( char* _base, int length )
        {
            //-1-
            int digitCount       = 0,
                upperLetterCount = 0,
                hyphenCount      = 0, punctuationCount = 0,
                lowerLetterCount = 0,
                dotCount         = 0,
                romanNumberCount = 0;
            var hasUmlautes = default(ExtraWordType);

            //-2-
            #region [.main cycle.]
            for ( int i = 0; i < length; i++ )
            {
                var ch = *(_base + i);
                var ct = *(_CTM  + ch);
                if ( (ct & CharType.IsDigit) == CharType.IsDigit )
                {
                    digitCount++;
                }
                else if ( (ct & CharType.IsLower) == CharType.IsLower )
                {
                    lowerLetterCount++;
                    if ( UmlautesNormalizer.IsUmlauteSymbol( ch ) )
                        hasUmlautes = ExtraWordType.HasUmlautes;
                }
                else if ( (ct & CharType.IsUpper) == CharType.IsUpper )
                {
                    upperLetterCount++;
                    if ( UmlautesNormalizer.IsUmlauteSymbol( ch ) )
                        hasUmlautes = ExtraWordType.HasUmlautes;
                    else if ( IsRomanSymbol( ch ) )
                        romanNumberCount++;
                }
                else if ( (ct & CharType.IsHyphen) == CharType.IsHyphen )
                {
                    hyphenCount++;
                    punctuationCount++;
                }
                else if ( xlat.IsDot( ch ) )
                {
                    dotCount++;
                    punctuationCount++;
                }
                else if ( (ct & CharType.IsPunctuation) == CharType.IsPunctuation )
                {
                    punctuationCount++;
                }
            }
            #endregion

            var first_ch = *_base;
            var first_ct = *(_CTM + first_ch);
            //-3-
            var isFirstUpper = (1 < length) && ((first_ct & CharType.IsUpper) == CharType.IsUpper);
            if ( (dotCount == 0) && (digitCount != 0) )
            {
                if ( isFirstUpper )
                    return (PosTaggerInputType.NumCapital, hasUmlautes);

                //'3G', '3-GMS', '123/Xyz'
                if ( (1 < length) && (upperLetterCount != 0) )
                {
                    for ( int i = 1; i < length; i++ )
                    {
                        var ch = *(_base + i);
                        var ct = *(_CTM + ch);
                        if ( (ct & CharType.IsUpper) == CharType.IsUpper )
                        {
                            return (PosTaggerInputType.NumCapital, hasUmlautes);
                        }
                        else
                        if ( (ct & CharType.IsLower) == CharType.IsLower )
                        {
                            break;    
                        }
                    }
                }
            }

            if ( upperLetterCount != 0 )
            {
                //(allC), (allCP), (mixCP) - все заглавные буквы на латинице (больше одной)
                if ( (1 < upperLetterCount) )
                {
                    if ( dotCount == 0 )
                    {
                        //(latC) - хотя бы одна римская цифра буква (без точки), не содержит арабских цифр, между буквами может быть тире: XXI;
                        if ( (romanNumberCount == length) || (romanNumberCount + hyphenCount == length) )
                        {
                            return (PosTaggerInputType.LatinCapital, hasUmlautes);
                        }

                        //(allC)  - все заглавные буквы на латинице (больше одной), нет точек, не содержит цифр, между буквами может быть тире : NATO;
                        if ( (upperLetterCount == length) || (upperLetterCount + hyphenCount == length) )
                        {
                            return (PosTaggerInputType.AllCapital, hasUmlautes);
                        }
                    }
                    else
                    {
                        //(allCP) - все заглавные буквы (больше одной) с точкой (точками), без тире: U.N.;
                        if ( (upperLetterCount + dotCount == length) && (hyphenCount == 0) )
                        {
                            return (PosTaggerInputType.AllCapitalWithDot, hasUmlautes);
                        }

                        //(mixCP) - две и более заглавных подряд  с точкой (точками). Между буквами может быть тире: V.IVI.PA;
                        if ( (upperLetterCount + dotCount == length) || (upperLetterCount + dotCount + hyphenCount == length) )
                        {
                            return (PosTaggerInputType.MixCapitalWithDot, hasUmlautes);
                        }
                    }
                }

                //(latC) - хотя бы одна римская цифра буква (без точки), не содержит арабских цифр, между буквами может быть тире: XXI;
                if ( (dotCount == 0) && ((romanNumberCount == length) || (romanNumberCount + hyphenCount == length)) )
                {
                    return (PosTaggerInputType.LatinCapital, hasUmlautes);
                }

                //(oneC) - одна заглавная буква без точки и не цифра: F;
                if ( (upperLetterCount == 1) && (length == 1) )
                {
                    return (PosTaggerInputType.OneCapital, hasUmlautes);
                }

                //(oneCP) - одна заглавная буква с точкой, перед буквой может быть тире: F. ;
                if ( dotCount == 1 )
                {
                    switch ( length )
                    {
                        case 2:
                            if ( (first_ct & CharType.IsUpper) == CharType.IsUpper )
                            {
                                return (PosTaggerInputType.OneCapitalWithDot, hasUmlautes);
                            }
                        break;

                        case 3:
                            if ( (first_ct             & CharType.IsHyphen) == CharType.IsHyphen &&
                                 (_CTM[ *(_base + 1) ] & CharType.IsUpper ) == CharType.IsUpper 
                               )
                            {
                                return (PosTaggerInputType.OneCapitalWithDot, hasUmlautes);
                            }
                        break;
                    }
                }


                //(mixC), (Z) - начинается с заглавной буквы
                if ( (first_ct & CharType.IsUpper) == CharType.IsUpper )
                {
                    //(Z) - только первая заглавная: Thatcher;
                    if ( (upperLetterCount == 1) && (lowerLetterCount + 1 == length) )
                    {
                        return (PosTaggerInputType.LatinFirstCapital, hasUmlautes);
                    }

                    //(mixC) - смешенные буквы: заглавные и строчные, первая буква - заглавная, между буквами может быть тире: St.-Petersburg , FireFox, Google.Maps
                    return (PosTaggerInputType.MixCapital, hasUmlautes);
                }

                //(iProd) - первые строчные латиница без точки + заглавная, между буквами может быть тире: iPod.
                if ( (first_ct & CharType.IsLower) == CharType.IsLower )
                {
                    if ( (digitCount == 0) && (dotCount == 0) )
                    {
                        return (PosTaggerInputType.FirstLowerWithUpper, hasUmlautes);
                    }
                }
            }

            //(Q) - кавычки ["«“”»]
            if ( (first_ct & CharType.IsQuote) == CharType.IsQuote )
            {
                return (PosTaggerInputType.Quote, ExtraWordType.Punctuation | hasUmlautes);
            }

            if ( (lowerLetterCount == 0) && (upperLetterCount == 0) )
            {
                //(NUM) - цифры в любой комбинации со знаками препинаний без букв: ["2,4", "10000", "2.456.542", "8:45"]
                if ( digitCount != 0 )
                {
                    var extraWordType = (length == digitCount) ? ExtraWordType.IntegerNumber : ExtraWordType.Other;
                    return (PosTaggerInputType.Num, extraWordType | hasUmlautes);
                }
            }

            if ( punctuationCount != 0 )
            {
                ExtraWordType extraWordType;
                if ( length == 1 )
                {
                    if ( (first_ct & CharType.IsHyphen) == CharType.IsHyphen )
                    {
                        extraWordType = ExtraWordType.Dash;
                    }
                    else
                    {
                        switch ( first_ch )
                        {
                            case ':': extraWordType = ExtraWordType.Colon; break;
                            case ',': extraWordType = ExtraWordType.Comma; break;
                            default : extraWordType = ExtraWordType.Other; break;
                        }
                    }
                }
                else if ( (lowerLetterCount != 0) || (upperLetterCount != 0) )
                {
                    return (PosTaggerInputType.Other, ExtraWordType.Other | hasUmlautes);
                }
                else
                {
                    extraWordType = ExtraWordType.Other;
                }
                return (PosTaggerInputType.Other, extraWordType | ExtraWordType.Punctuation | hasUmlautes);
            }

            return (PosTaggerInputType.Other, hasUmlautes);
        }
        [M(O.AggressiveInlining)] unsafe public (PosTaggerInputType posTaggerInputType, ExtraWordType extraWordType) GetPosTaggerInputType( word_t word )
        {
            fixed ( char* _base = word.valueOriginal )
            {
                return (GetPosTaggerInputType( _base, word.valueOriginal.Length ));
            }
        }
    }
}
