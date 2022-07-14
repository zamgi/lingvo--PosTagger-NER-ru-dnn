using System;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    [Flags] public enum ExtraWordType : byte
    {
        Other = 0, // other's (другой)

        //skip-ignore url's        
        Comma         = 1,        // – запятая;
        Dash          = (1 << 1), // – тире;
        Colon         = (1 << 2), // – двоеточие;
        IntegerNumber = (1 << 3), // – содержит хотя бы одну цифру и не содержит букв;
        //OneCapital,    // - первая заглавная с точкой;
        //FirstCapital,  // - первая заглавная, не содержит пробелов;
        //ComplexPhrase, // - составные (имеющие хотя бы один пробел);

        //Abbreviation,
        Punctuation = (1 << 4),
        //---Url,    // – все url & e-mail;

        HasUmlautes = (1 << 5),
    }
}
