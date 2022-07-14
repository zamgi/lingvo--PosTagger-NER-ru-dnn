namespace Lingvo.PosTagger.WebService
{
    /// <summary>
    /// 
    /// </summary>
    internal static class WebApiConsts
    {      
        /// <summary>
        /// 
        /// </summary>
        internal static class PosTagger
        {
            public const string RoutePrefix = "PosTagger";

            public const string Run              = "Run";
            public const string GetModelInfoKeys = "GetModelInfoKeys";
            public const string ReadLogFile      = "ReadLogFile";
            public const string Log              = "Log";
            public const string DeleteLogFile    = "DeleteLogFile";
            public const string DelLog           = "DelLog";
        }
    }
}
