using System.Collections.Generic;

using Lingvo.PosTagger.Applications;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class Predictor
    {
        private SeqLabel _SL;

        public Predictor( string modelFilePath, int maxPredictSentLength, ProcessorTypeEnums processorType, string deviceIds )
        {
            var opts = new Options()
            {
                ModelFilePath        = modelFilePath,
                MaxPredictSentLength = maxPredictSentLength,
                ProcessorType        = processorType,
                DeviceIds            = deviceIds,
            };
            _SL = SeqLabel.Create4Predict( opts );
        }
        public Predictor( Options opts ) => _SL = SeqLabel.Create4Predict( opts );
        public Predictor( SeqLabel sl ) => _SL = sl;

        public List< string > Predict_LabelTokens( List< string > inputTokens, int? maxPredictSentLength = null, float cutDropout = 0.1f ) => _SL.Predict_Full( inputTokens, maxPredictSentLength, cutDropout ).labelTokens;
        public (List< string > labelTokens, NetworkResult.ClassesInfo classesInfos) Predict( List< string > inputTokens, int? maxPredictSentLength = null, float cutDropout = 0.1f ) => _SL.Predict_Full( inputTokens, maxPredictSentLength, cutDropout, returnWordClassInfos: true );
        public NetworkResult.ClassesInfo Predict_ClassesInfo( List< string > inputTokens, int? maxPredictSentLength = null, float cutDropout = 0.1f ) => _SL.Predict_Full( inputTokens, maxPredictSentLength, cutDropout, returnWordClassInfos: true ).classesInfos;
    }
}

