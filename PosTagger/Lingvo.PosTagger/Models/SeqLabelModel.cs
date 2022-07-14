using System.Linq;

using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class SeqLabelModel : Model
    {
        public SeqLabelModel() { }
        public SeqLabelModel( int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, Vocab tgtVocab )
            : base( hiddenDim, encoderLayerDepth, encoderType, embeddingDim, multiHeadNum, srcVocab/*, applyContextEmbeddingsToEntireSequence: false*/ )
        {
            TgtVocab = tgtVocab;
        }
        public SeqLabelModel( Model_4_ProtoBufSerializer m )
            : base( m.HiddenDim, m.EncoderLayerDepth, m.EncoderType, m.EncoderEmbeddingDim, m.MultiHeadNum, m.SrcVocab?.ToVocab()/*, applyContextEmbeddingsToEntireSequence: false*/ )
        {
            //---ClsVocabs        = m.ClsVocabs?.Select( v => v.ToVocab() ).ToList();
            TgtVocab         = m.ClsVocabs?.Select( v => v.ToVocab() ).ToList().FirstOrDefault() ?? m.TgtVocab.ToVocab();
            __Name2Weights   = m.Name2Weights;
            BestPrimaryScore = m.BestPrimaryScores.AnyEx() ? m.BestPrimaryScores.Values.First() : default;
        }
        public static SeqLabelModel Create( Model_4_ProtoBufSerializer m ) => new SeqLabelModel( m );
    }
}
