using System.Collections.Generic;

namespace Lingvo.PosTagger.Metrics
{
    /// <summary>
    /// 
    /// </summary>
    public interface IMetric
    {
        void Evaluate( List< List< string > > allRefTokens, List< string > hypTokens );
        void Evaluate( List< string > refTokens, List< string > hypTokens );
        void ClearStatus();
        string Name { get; }
        string GetScoreStr();
        double GetPrimaryScore();
        (double primaryScore, string text) GetScore();
    }
}
