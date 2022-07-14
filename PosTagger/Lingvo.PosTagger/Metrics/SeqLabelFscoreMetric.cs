using System;
using System.Collections.Generic;

namespace Lingvo.PosTagger.Metrics
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class SeqLabelFscoreMetric : IMetric
    {
        private readonly string _ClassLabel;
        private double[] _Counts;
        public SeqLabelFscoreMetric( string classLabel )
        {
            _Counts     = new double[ 3 ];
            _ClassLabel = classLabel;
        }

        public string Name => _ClassLabel;

        public void ClearStatus() => Array.Clear( _Counts, 0, _Counts.Length );

        public void Evaluate( List< List< string > > allRefTokens, List< string > hypTokens )
        {
            foreach ( List< string > refTokens in allRefTokens )
            {
                for ( int i = 0, len_2 = Math.Min( refTokens.Count, hypTokens.Count ); i < len_2; i++ )
                {
                    var hypToken = hypTokens[ i ];
                    var refToken = refTokens[ i ];

                    var eq_1 = (hypToken == _ClassLabel);
                    var eq_2 = (refToken == _ClassLabel);
                    if ( eq_1 ) _Counts[ 1 ]++;
                    if ( eq_2 ) _Counts[ 2 ]++;
                    if ( eq_1 && eq_2 ) _Counts[ 0 ]++;
                }
            }
        }
        public void Evaluate( List< string > refTokens, List< string > hypTokens )
        {
            for ( int i = 0, len = Math.Min( refTokens.Count, hypTokens.Count ); i < len; i++ )
            {
                var hypToken = hypTokens[ i ];
                var refToken = refTokens[ i ];

                var eq_1 = (hypToken == _ClassLabel);
                var eq_2 = (refToken == _ClassLabel);
                if ( eq_1 ) _Counts[ 1 ]++;
                if ( eq_2 ) _Counts[ 2 ]++;
                if ( eq_1 && eq_2 ) _Counts[ 0 ]++;
            }
        }

        public string GetScoreStr()
        {
            if ( _Counts[ 1 ] == 0.0 || _Counts[ 2 ] == 0.0 )
            {
                return ($"No F-score available for '{_ClassLabel}'");
            }

            double precision = _Counts[ 0 ] / _Counts[ 1 ];
            double recall = _Counts[ 0 ] / _Counts[ 2 ];
            double objective = 0.0;
            if ( precision > 0.0 && recall > 0.0 )
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return ($"F-score = '{100.0 * objective:F}' Precision = '{100.0 * precision:F}' Recall = '{100.0 * recall:F}'");
        }
        public double GetPrimaryScore()
        {
            if ( _Counts[ 1 ] == 0.0 || _Counts[ 2 ] == 0.0 )
            {
                return (0.0);
            }

            double precision = _Counts[ 0 ] / _Counts[ 1 ];
            double recall    = _Counts[ 0 ] / _Counts[ 2 ];
            double objective = 0.0;
            if ( precision > 0.0 && recall > 0.0 )
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }
            return (100.0 * objective);
        }
        public (double primaryScore, string text) GetScore()
        {
            if ( _Counts[ 1 ] == 0.0 || _Counts[ 2 ] == 0.0 )
            {
                return (0.0, $"No F-score available for '{_ClassLabel}'");
            }

            double precision = _Counts[ 0 ] / _Counts[ 1 ];
            double recall    = _Counts[ 0 ] / _Counts[ 2 ];
            double objective = 0.0;
            if ( precision > 0.0 && recall > 0.0 )
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return (100.0 * objective, $"F-score = '{100.0 * objective:F}' Precision = '{100.0 * precision:F}' Recall = '{100.0 * recall:F}'");
        }
    }
}
