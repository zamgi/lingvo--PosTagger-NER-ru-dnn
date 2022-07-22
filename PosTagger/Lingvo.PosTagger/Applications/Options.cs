using Lingvo.PosTagger.Text;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Applications
{
    /// <summary>
    /// 
    /// </summary>
    public readonly struct OptionsAllowedChanging
    {
        /// <summary>
        /// 
        /// </summary>
        public delegate void ChangingDelege( in OptionsAllowedChanging opts );

        #region [.2.]
        public int BatchSize             { get; init; }
        public int MaxEpochNum           { get; init; }
        public int Valid_RunEveryUpdates { get; init; }
        #endregion
    }

    /// <summary>
    /// 
    /// </summary>
    public class Options
    {
        #region [.1.]
        [Arg(nameof(EmbeddingDim), "The embedding dimension. (Default = 128).")] public int EmbeddingDim = 128;

        [Arg(nameof(MaxPredictSentLength), "Maxmium sentence length in valid and predict set. (Default = 220).")] public int MaxPredictSentLength = 220;
        [Arg(nameof(MaxTrainSentLength)  , "Maxmium sentence length in training corpus. (Default = 220)."      )] public int MaxTrainSentLength   = 220;

        [Arg(nameof(SentSplitterResourcesXmlFilename))] public string SentSplitterResourcesXmlFilename;
        [Arg(nameof(UrlDetectorResourcesXmlFilename) )] public string UrlDetectorResourcesXmlFilename;
        [Arg(nameof(MaxEndingLength))] public int MaxEndingLength;

        [Arg(nameof(CurrentDirectory))] public string CurrentDirectory;
        [Arg(nameof(ConfigFilePath), "The file path of config file for parameters.")] public string ConfigFilePath;

        /// <summary>
        /// 
        /// </summary>
        public struct ExternalValidator_t
        {
            public string FileName;
            public string Arguments;
            public string WorkingDirectory;
        }
        [Arg(nameof(ExternalValidator)     )] public ExternalValidator_t ExternalValidator;
        [Arg(nameof(TryRunValidateParallel))] public bool TryRunValidateParallel;
        #endregion

        #region [.2.]
        [Arg(nameof(ModelFilePath), "The trained model file path. (Default = \"model.s2s\").")] public string ModelFilePath = "model.s2s";

        [Arg(nameof(BatchSize), "The batch size. (Default = 1).")] public int BatchSize = 1;

        [Arg(nameof(MaxEpochNum), "Maxmium epoch number during training. (Default = 100).")] public int MaxEpochNum = 100;

        [Arg(nameof(TrainCorpusPath), "Training corpus folder path.")] public string TrainCorpusPath;
        [Arg(nameof(ValidCorpusPath), "Valid corpus folder path."   )] public string ValidCorpusPath;

        [Arg(nameof(InputTestFile) , "The test input file." ) ] public string InputTestFile;
        [Arg(nameof(OutputTestFile), "The test output file.")] public string OutputTestFile;
        [Arg(nameof(ValidationOutputFileName))] public string ValidationOutputFileName;

        //[Arg(nameof(ValidIntervalHours), "The interval hours to run model validation")] public float ValidIntervalHours = 1.0f;
        [Arg(nameof(Valid_StartAfterUpdates), "Start to run validation after N updates. (Default = 20 000).")] public int Valid_StartAfterUpdates = 20_000;
        [Arg(nameof(Valid_RunEveryUpdates)  , "Run validation every certain updates. (Default = 10 000)."   )]public int Valid_RunEveryUpdates = 10_000;

        [Arg(nameof(SrcVocabSize), "The size of vocabulary in source side. (Default = 50 000).")] public int SrcVocabSize = 50_000;
        //[Arg(nameof(TgtVocabSize), "The size of vocabulary in target side. (Default = 50 000).")] public int TgtVocabSize = 50_000;        


        [Arg(nameof(SrcVocab), "The vocabulary file path for source side.")] public string SrcVocab;
        [Arg(nameof(TgtVocab), "The vocabulary file path for target side.")] public string TgtVocab;
        #endregion

        #region [.3.]
        [Arg(nameof(ProcessorType), "Processor type: GPU, CPU. (Default = GPU).")]
        public ProcessorTypeEnums ProcessorType = ProcessorTypeEnums.GPU;

        [Arg(nameof(DeviceIds), "Device ids for training in GPU mode. (Default = 0. For multi devices, ids are split by comma, for example: 0,1,2).")]
        public string DeviceIds = "0";

        [Arg(nameof(CompilerOptions), "The options for CUDA NVRTC compiler. Options are split by space. (For example: \"--use_fast_math --gpu-architecture=compute_60\"). (Default = \"--use_fast_math\").")]
        public string CompilerOptions = "--use_fast_math";


        [Arg(nameof(EncoderType), "Encoder type: BiLSTM, Transformer. (Default = Transformer).")]
        public EncoderTypeEnums EncoderType = EncoderTypeEnums.Transformer;

        [Arg(nameof(HiddenSize), "The hidden layer size of encoder and decoder. (Default = 128).")]
        public int HiddenSize = 128;
        
        [Arg(nameof(EncoderLayerDepth), "The network depth in encoder. (Default = 1).")]
        public int EncoderLayerDepth = 1;

        [Arg(nameof(MultiHeadNum), "The number of multi-heads in transformer model. (Default = 8).")]
        public int MultiHeadNum = 8;


        [Arg(nameof(OptimizerType), "The weights optimizer for training: Adam, RMSProp. (Default = Adam).")]
        public OptimizerTypeEnums OptimizerType = OptimizerTypeEnums.Adam;

        [Arg(nameof(Beta1)   , "The beta1 for optimizer. (Default = 0.9)." )] public float Beta1    = 0.9f;
        [Arg(nameof(Beta2)   , "The beta2 for optimizer. (Default = 0.98).")] public float Beta2    = 0.98f;
        [Arg(nameof(GradClip), "Clip gradients. (Default = 3.0)."          )] public float GradClip = 3.0f;

        
        [Arg(nameof(DropoutRatio), "Dropout ratio. (Default = 0).")]
        public float DropoutRatio = 0.0f;

        [Arg(nameof(EncoderStartLearningRateFactor), "Starting Learning rate factor for encoders. (Default = 1).")]
        public float EncoderStartLearningRateFactor = 1.0f;

        [Arg(nameof(IsEncoderTrainable), "It indicates if the encoder is trainable. (Default = true).")]
        public bool IsEncoderTrainable = true;

        [Arg(nameof(MemoryUsageRatio), "The ratio of memory usage. (Default = 0.95).")]
        public float MemoryUsageRatio = 0.95f;

        [Arg(nameof(StartLearningRate), "Starting Learning rate. (Default = 0.0006).")]
        public float StartLearningRate = 0.0006f;

        [Arg(nameof(TooLongSequence), "How to deal with too long sequence: Ignore, Truncation. (Default = Ignore).")]
        public TooLongSequence TooLongSequence = TooLongSequence.Ignore;

        [Arg(nameof(UpdateFreq), "Update parameters every N batches. (Default = 1).")]
        public int UpdateFreq = 1;

        [Arg(nameof(WarmUpSteps), "The number of steps for warming up. (Default = 8000).")]
        public int WarmUpSteps = 8000;

        [Arg(nameof(WeightsUpdateCount), "The number of updates for weights.")]
        public int WeightsUpdateCount;
        #endregion
    }
}
