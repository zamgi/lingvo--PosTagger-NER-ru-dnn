namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    public enum PosTaggerInputType : byte
    {
        #region [.common.]
        Other,              // other's (другой)
        AllCapital,         // Все заглавные буквы (больше одной) [МТС]        
        LatinCapital,       // Только первая заглавная на латинице [Fox]
        MixCapital,         // Смешенные заглавные и прописные буквы; 
                               //русский   : {латиница + кириллица [СевКавГПУ]}, 
                               //английский: {заглавные и строчные, первая буква - заглавная, между буквами может быть тире, точка: St.-Petersburg , FireFox, Google.Maps}
        MixCapitalWithDot,  // Все заглавные буквы (больше одной) подряд с точкой (точками) [V.IV.I.PA]
        NumCapital,         // Начинается с заглавной буквы и содержит хотябы одну цифру [МИГ-21]
        OneCapital,         // Одна заглавная буква без точки [F]
        OneCapitalWithDot,  // одна заглавная буква с точкой [F.]
        FirstLowerWithUpper,// первая буква строчная; в слове нет точек; обязательно присутствует заглавная буква
        Quote,              // кавычки ["«“”»]
        Num,                // цифры в любой комбинации со знаками препинаний без букв [2,4 ; 10000 ; 2.456.542 ; 8:45]
        #endregion

        #region [.russian-language.]
        AllLatinCapital, // все буквы заглавные и все на латинице [POP]
        LatinNum,        // Хотя бы одна римская цифра буква (без точки) [XVI] [X-XI]        
        FirstCapital,    // Только первая заглавная на кириллице [Вася]            
        Comma,           // запятую и точку с запятой
        #endregion

        #region [.english-language.]
        AllCapitalWithDot, // все заглавные буквы (больше одной) с точкой (точками), без тире: [U.N.]
        LatinFirstCapital, // только первая заглавная:  [Thatcher]
        #endregion

        Email,
        Url
    }

    /// <summary>
    /// 
    /// </summary>
    public static partial class PosTaggerExtensions
    {
        public static string ToText( this PosTaggerInputType posTaggerInputType ) => posTaggerInputType.ToString();  
    }
}
