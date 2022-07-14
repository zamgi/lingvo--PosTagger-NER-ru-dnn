using System.Collections.Generic;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.WebService
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class Config : Options
    {
        /// <summary>
        /// 
        /// </summary>
        public struct ModelInfo
        {
            public string ModelFilePath;
            //public bool   DelayLoad;
            public bool   LoadImmediate;
            public bool   DontReplaceNumsOnPlaceholders;
            public int    MaxEndingLength;
            public string RegimenModelType;
        }

        [Arg(nameof(CONCURRENT_FACTORY_INSTANCE_COUNT))] public int? CONCURRENT_FACTORY_INSTANCE_COUNT;
        [Arg(nameof(ModelInfos))] public Dictionary< string, ModelInfo > ModelInfos;

        /// <summary>
        /// 
        /// </summary>
        public struct LogToFile
        {
            public bool   Enable;
            public string LogFileName;
        }
        [Arg(nameof(Log))] public LogToFile Log;
    }
}