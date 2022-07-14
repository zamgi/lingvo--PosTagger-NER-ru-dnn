using System.Collections.Generic;
using System.Linq;

using Lingvo.PosTagger.Utils;

using ProtoBuf;

namespace Lingvo.PosTagger.Models
{
    /// <summary>
    /// 
    /// </summary>
    [ProtoContract(SkipConstructor=true), ProtoInclude(100, typeof(Vocab_4_ProtoBufSerializer)),
                                          ProtoInclude(101, typeof(DecoderTypeEnums)),
                                          ProtoInclude(101, typeof(EncoderTypeEnums))]
    public sealed class Model_4_ProtoBufSerializer
    {
        public Model_4_ProtoBufSerializer() { }
        public Model_4_ProtoBufSerializer( Model m )
        {
            Name2Weights        = m.GetName2Weights();
            EncoderEmbeddingDim = m.EncoderEmbeddingDim;
            EncoderLayerDepth   = m.EncoderLayerDepth;
            EncoderType         = m.EncoderType;
            HiddenDim           = m.HiddenDim;
            MultiHeadNum        = m.MultiHeadNum;
            SrcVocab            = new Vocab_4_ProtoBufSerializer( m.SrcVocab );
            TgtVocab            = new Vocab_4_ProtoBufSerializer( m.TgtVocab ); //(m.TgtVocab != null) ? new Vocab_4_ProtoBufSerializer( m.TgtVocab ) : null;            
            BestPrimaryScores   = new Dictionary< string, double > { { string.Empty, m.BestPrimaryScore } };
            //ClsVocabs           = m.ClsVocabs?.Select( c => new Vocab_4_ProtoBufSerializer( c ) ).ToList();
            //ApplyContextEmbeddingsToEntireSequence = m.ApplyContextEmbeddingsToEntireSequence;
        }
        public static Model_4_ProtoBufSerializer Create( Model m ) => new Model_4_ProtoBufSerializer( m );

        [ProtoMember(1)] public Dictionary< string, float[] > Name2Weights { get; set; }
        [ProtoMember(2)] public int DecoderEmbeddingDim { get; set; } //NOT USED//
        [ProtoMember(3)] public int EncoderEmbeddingDim { get; set; }
        [ProtoMember(4)] public int DecoderLayerDepth { get; set; } //NOT USED//
        [ProtoMember(5)] public int EncoderLayerDepth { get; set; }
        [ProtoMember(6)] public DecoderTypeEnums DecoderType { get; set; } //NOT USED//
        [ProtoMember(7)] public EncoderTypeEnums EncoderType { get; set; }
        [ProtoMember(8)] public int HiddenDim { get; set; }
        [ProtoMember(9)] public bool EnableSegmentEmbeddings { get; set; } //NOT USED//
        [ProtoMember(10)] public int MultiHeadNum { get; set; }
        [ProtoMember(11)] public Vocab_4_ProtoBufSerializer SrcVocab { get; set; }
        [ProtoMember(12)] public Vocab_4_ProtoBufSerializer TgtVocab { get; set; }
        [ProtoMember(13)] public List< Vocab_4_ProtoBufSerializer > ClsVocabs { get; set; }
        [ProtoMember(14)] public bool EnableCoverageModel { get; set; } //NOT USED//
        [ProtoMember(15)] public bool SharedEmbeddings { get; set; } //NOT USED//
        [ProtoMember(16)] public string SimilarityType { get; set; } //NOT USED//
        [ProtoMember(17)] public bool ApplyContextEmbeddingsToEntireSequence { get; set; }
        [ProtoMember(19)] public int MaxSegmentNum { get; set; } //NOT USED//
        [ProtoMember(20)] public bool PointerGenerator { get; set; } //NOT USED//

        [ProtoMember(21)] public Dictionary< string, double > BestPrimaryScores { get; set; }
    }
}
