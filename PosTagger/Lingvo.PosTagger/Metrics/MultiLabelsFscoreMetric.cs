using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Metrics
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class MultiLabelsFscoreMetric : IMetric
    {
        private (string classLabel, double[] cnts/*, double , double , double */)[] _ClassInfo;
        private string _GroupName; 

        public MultiLabelsFscoreMetric( IEnumerable< string > classLabels, string groupName = null )
        {
            _ClassInfo = classLabels.Select( classLabel => (classLabel, new double[ 3 ]) ).ToArray( /*classLabels.Count*/ );
            _GroupName = groupName;

            Logger.WriteLine( $"Added '{string.Join( "', '", classLabels )}' ({_ClassInfo.Length}) labels to '{Name}'." );
        }

        public string Name => (_GroupName != null) ? $"MultiLabelsFscore '{_GroupName}'" : "MultiLabelsFscore";

        public void ClearStatus()
        {
            for ( int i = _ClassInfo.Length - 1; 0 <= i; i-- )
            {
                var cnts = _ClassInfo[ i ].cnts;
                Array.Clear( cnts, 0, cnts.Length );

                //ref var ptr_t = ref _ClassInfo[ i ];
                //Array.Clear( ptr_t.cnts, 0, ptr_t.cnts.Length );
            }
        }

        public void Evaluate( List< List< string > > allRefTokens, List< string > hypTokens )
        {
            for ( int j = 0, len = _ClassInfo.Length; j < len; j++ )
            {
                ref var t = ref _ClassInfo[ j ];
                var cnts       = t.cnts;
                var classLabel = t.classLabel;
                foreach ( List< string > refTokens in allRefTokens )
                {
                    try
                    {
                        for ( int i = 0, len_2 = Math.Min( refTokens.Count, hypTokens.Count ); i < len_2; i++ )
                        {
                            var hypToken = hypTokens[ i ];
                            var refToken = refTokens[ i ];

                            var eq_1 = (hypToken == classLabel);
                            var eq_2 = (refToken == classLabel);
                            if ( eq_1 ) cnts[ 1 ]++;
                            if ( eq_2 ) cnts[ 2 ]++;
                            if ( eq_1 && eq_2 ) cnts[ 0 ]++;
                        }
                    }
                    catch ( Exception ex )
                    {
                        Logger.WriteLine( $"Exception: {ex.Message}, Ref = '{string.Join( " ", refTokens )}', Hyp = '{string.Join( " ", hypTokens )}'" );
                        throw;
                    }
                }
            }
        }
        public void Evaluate( List< string > refTokens, List< string > hypTokens )
        {
            for ( int j = 0, len = _ClassInfo.Length; j < len; j++ )
            {
                ref var t = ref _ClassInfo[ j ];
                var cnts       = t.cnts;
                var classLabel = t.classLabel;
                
                try
                {
                    for ( int i = 0, len_2 = Math.Min( refTokens.Count, hypTokens.Count ); i < len_2; i++ )
                    {
                        var hypToken = hypTokens[ i ];
                        var refToken = refTokens[ i ];

                        var eq_1 = (hypToken == classLabel);
                        var eq_2 = (refToken == classLabel);
                        if ( eq_1 ) cnts[ 1 ]++;
                        if ( eq_2 ) cnts[ 2 ]++;
                        if ( eq_1 && eq_2 ) cnts[ 0 ]++;
                    }
                }
                catch ( Exception ex )
                {
                    Logger.WriteLine( $"Exception: {ex.Message}, Ref = '{string.Join( " ", refTokens )}', Hyp = '{string.Join( " ", hypTokens )}'" );
                    throw;
                }
            }
        }

        public string GetScoreStr()
        {
            var buf = new StringBuilder( _ClassInfo.Length * 100 );
            var max_len = _ClassInfo.Any() ? _ClassInfo.Max( t => t.classLabel.Length ) : 0;
            var clsCnt  = 0;
            for ( int i = 0, len = _ClassInfo.Length; i < len; i++ )
            {
                ref var t = ref _ClassInfo[ i ];
                var cnts       = t.cnts;
                var classLabel = t.classLabel;
                if ( cnts[ 1 ] == 0.0 || cnts[ 2 ] == 0.0 )
                {
                    continue;
                }

                double precision = cnts[ 0 ] / cnts[ 1 ];
                double recall    = cnts[ 0 ] / cnts[ 2 ];
                double objective = 0.0;
                if ( 0.0 < precision && 0.0 < recall )
                {
                    objective = 2.0 * (precision * recall) / (precision + recall);
                }

                buf.AppendLine( $"{classLabel.PadRight( max_len, ' ' )}: F-score = '{100.0 * objective:F}' Precision = '{100.0 * precision:F}' Recall = '{100.0 * recall:F}'" );
                clsCnt++;
            }
            return ($"Common-Score: '{GetPrimaryScore():F}'\n" + string.Join( "\n", buf.ToString() ) + $"\nThe number of categories = '{clsCnt}' of '{_ClassInfo.Length}'\n");
        }
        public double GetPrimaryScore()
        {
            double score = 0.0;
            for ( int i = _ClassInfo.Length - 1; 0 <= i; i-- )
            {
                var cnts = _ClassInfo[ i ].cnts;
                if ( cnts[ 1 ] == 0.0 || cnts[ 2 ] == 0.0 )
                {
                    //score += 0.0;
                    continue;
                }

                double precision = cnts[ 0 ] / cnts[ 1 ];
                double recall    = cnts[ 0 ] / cnts[ 2 ];
                double objective = 0.0;
                if ( 0.0 < precision && 0.0 < recall )
                {
                    objective = 2.0 * (precision * recall) / (precision + recall);
                }

                score += 100.0 * objective;
            }

            return (score / _ClassInfo.Length );
        }
        public (double primaryScore, string text) GetScore() => (GetPrimaryScore(), GetScoreStr());
    }
}
