using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tokenizing
{
    /// <summary>
    /// 
    /// </summary>
    public class word_t
    {
        public string valueOriginal;
        public string valueUpper;
        public string valueUpper__UmlautesNormalized;
        public int    startIndex;
        public int    length;
        public ExtraWordType extraWordType;
//#if DEBUG
//        [M(O.AggressiveInlining)] public word_t() { }
//#endif
        [M(O.AggressiveInlining)] public int endIndex() => startIndex + length;

        #region [.PosTagger.]
        public PosTaggerInputType  posTaggerInputType;
        public PosTaggerOutputType posTaggerOutputType;
        #endregion

        #region [.Raw SeqLabel output-type.]
        public string seqLabelOutputType;
        #endregion

        #region [.to-string's.]
        public string GetOutputType()
        {
            var p = posTaggerOutputType.ToText();
            return ((seqLabelOutputType != null) && (seqLabelOutputType != p) ? $"{p} ({seqLabelOutputType})" : p);
        }
        public override string ToString()
        {
            var p = posTaggerOutputType.ToText();
            var t = ((seqLabelOutputType != null) && (seqLabelOutputType != p) ? $" ({seqLabelOutputType})" : null);
            return ($"{p}{t},  '{valueOriginal}', [{startIndex}:{length}]");
        }
        #endregion
    }
}