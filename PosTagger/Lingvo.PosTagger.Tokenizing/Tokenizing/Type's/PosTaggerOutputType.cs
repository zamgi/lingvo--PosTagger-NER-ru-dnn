using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    public enum PosTaggerOutputType : byte
    {
        Other = 0,

        Noun,                // - Существительное 
        Adjective,           // - Прилагательное 
        AdjectivePronoun,    // -  Местоимённое прилагательное  (который, твой)
        PossessivePronoun,   // -  Притяжательное местоимение (свой, чей)
        Pronoun,             // -  Местоимение 
        Numeral,             //  - Числительное 
        Verb,                // -  Глагол 
        Infinitive,          // -  Инфинитив 
        Adverb,              // - Наречие 
        AdverbialParticiple, // -  Деепричастие 
        AdverbialPronoun,    // -  Местоимённое наречие (где, вот)
        Participle,          // -  Причастие 
        Conjunction,         // - Союз 
        Preposition,         // - Предлог 
        Interjection,        // -  Междометие (увы, батюшки)
        Particle,            // - Частица (бы, же, пусть)
        Article,             // - Артикль (в русском нет)
        AuxiliaryVerb,       // - Вспомогательный глагол 
        Predicative,         // - Предикатив (включает в себя модальные глаголы) (жаль, хорошо, пора)
        Punctuation,

        Email,
        Url
    }

    /// <summary>
    /// 
    /// </summary>
    public static partial class PosTaggerExtensions
    {
        public static string ToText( this PosTaggerOutputType posTaggerOutputType )
        {
            switch ( posTaggerOutputType )    
            {
                case PosTaggerOutputType.Noun               : return "Noun";
                case PosTaggerOutputType.Adjective          : return "Adjective";
                case PosTaggerOutputType.AdjectivePronoun   : return "AdjectivePronoun";
                case PosTaggerOutputType.PossessivePronoun  : return "PossessivePronoun";
                case PosTaggerOutputType.Pronoun            : return "Pronoun";
                case PosTaggerOutputType.Numeral            : return "Numeral";
                case PosTaggerOutputType.Verb               : return "Verb";
                case PosTaggerOutputType.Infinitive         : return "Infinitive";
                case PosTaggerOutputType.Adverb             : return "Adverb";
                case PosTaggerOutputType.AdverbialParticiple: return "AdverbialParticiple";
                case PosTaggerOutputType.AdverbialPronoun   : return "AdverbialPronoun";
                case PosTaggerOutputType.Participle         : return "Participle";
                case PosTaggerOutputType.Conjunction        : return "Conjunction";
                case PosTaggerOutputType.Preposition        : return "Preposition";
                case PosTaggerOutputType.Interjection       : return "Interjection";
                case PosTaggerOutputType.Particle           : return "Particle";
                case PosTaggerOutputType.Article            : return "Article";
                case PosTaggerOutputType.AuxiliaryVerb      : return "AuxiliaryVerb";
                case PosTaggerOutputType.Predicative        : return "Predicative";

                case PosTaggerOutputType.Email        : return "Email";
                case PosTaggerOutputType.Url          : return "Url";
                case PosTaggerOutputType.Other        : return "Other";
                default                               : return (posTaggerOutputType.ToString());
            }
        }
        [M(O.AggressiveInlining)] public static PosTaggerOutputType ToPosTaggerOutputType( this string posTaggerOutputType )
        {
            if ( posTaggerOutputType != null )
            {
                switch ( posTaggerOutputType )    
                {
                    case "Noun"               : return PosTaggerOutputType.Noun;
                    case "Adjective"          : return PosTaggerOutputType.Adjective;
                    case "AdjectivePronoun"   : return PosTaggerOutputType.AdjectivePronoun;
                    case "PossessivePronoun"  : return PosTaggerOutputType.PossessivePronoun;
                    case "Pronoun"            : return PosTaggerOutputType.Pronoun;
                    case "Numeral"            : return PosTaggerOutputType.Numeral;
                    case "Verb"               : return PosTaggerOutputType.Verb;
                    case "Infinitive"         : return PosTaggerOutputType.Infinitive;
                    case "Adverb"             : return PosTaggerOutputType.Adverb;
                    case "AdverbialParticiple": return PosTaggerOutputType.AdverbialParticiple;
                    case "AdverbialPronoun"   : return PosTaggerOutputType.AdverbialPronoun;
                    case "Participle"         : return PosTaggerOutputType.Participle;
                    case "Conjunction"        : return PosTaggerOutputType.Conjunction;
                    case "Preposition"        : return PosTaggerOutputType.Preposition;
                    case "Interjection"       : return PosTaggerOutputType.Interjection;
                    case "Particle"           : return PosTaggerOutputType.Particle;
                    case "Article"            : return PosTaggerOutputType.Article;
                    case "AuxiliaryVerb"      : return PosTaggerOutputType.AuxiliaryVerb;
                    case "Predicative"        : return PosTaggerOutputType.Predicative;
                    case "Punctuation"        : return PosTaggerOutputType.Punctuation;    

                    case "Email" : return (PosTaggerOutputType.Email);
                    case "Url"   : return (PosTaggerOutputType.Url);
                    case "Other" : return PosTaggerOutputType.Other;
                }
            }
            return (PosTaggerOutputType.Other);
        }        
    }
}
