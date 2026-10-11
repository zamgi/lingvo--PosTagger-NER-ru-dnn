using ProtoBuf;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    [ProtoContract(SkipConstructor=true)]
    public enum EncoderTypeEnums
    {
        BiLSTM,
        Transformer
    }

    [ProtoContract(SkipConstructor=true)]
    /// <summary>
    /// NOT USED
    /// </summary>
    public enum DecoderTypeEnums
    {
        AttentionLSTM,
        Transformer
    }
}
