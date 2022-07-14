using System;
using System.Collections.Generic;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Metrics
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RougeMetric : IMetric
    {
        private readonly HashSet<string> _AllRefWords;
        private readonly HashSet<string> _AllHypWords;
        private readonly HashSet<string> _UnionLCSWords;
        public RougeMetric()
        {
            _AllRefWords   = new HashSet<string>();
            _AllHypWords   = new HashSet<string>();
            _UnionLCSWords = new HashSet<string>();
        }

        public string Name => "RougeL";

        public void ClearStatus()
        {
            _AllRefWords.Clear();
            _AllHypWords.Clear();
            _UnionLCSWords.Clear();
        }

        public void Evaluate( List< List< string > > allRefTokens, List< string > hypTokens )
        {
            var refTokens = allRefTokens[ 0 ];
            foreach ( var token in refTokens )
            {
                _AllRefWords.Add( token );
            }

            foreach ( var token in hypTokens )
            {
                _AllHypWords.Add( token );
            }

            var results = LCS( refTokens, hypTokens );
            foreach ( var r in results )
            {
                _UnionLCSWords.Add( r );
            }
        }
        public void Evaluate( List< string > refTokens, List< string > hypTokens )
        {
            foreach ( var token in refTokens )
            {
                _AllRefWords.Add( token );
            }

            foreach ( var token in hypTokens )
            {
                _AllHypWords.Add( token );
            }

            var results = LCS( refTokens, hypTokens );
            foreach ( var r in results )
            {
                _UnionLCSWords.Add( r );
            }
        }

        public double GetPrimaryScore()
        {
            double recall    = _UnionLCSWords.Count / (double) _AllRefWords.Count;
            double precision = _UnionLCSWords.Count / (double) _AllHypWords.Count;
            double objective = 0.0;
            if ( precision > 0.0 && recall > 0.0 )
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }
            return (objective);
        }
        public string GetScoreStr() => GetPrimaryScore().ToString("F");
        public (double primaryScore, string text) GetScore()
        {
            var score = GetPrimaryScore();
            return (score, score.ToString("F"));
        }

        private static List<string> LCS( List<string> x, List<string> y )
        {
            int m = x.Count;
            int n = y.Count;

            var L = new int[ m + 1, n + 1 ];

            // Following steps build L[m+1][n+1] in bottom up fashion. Note 
            // that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]  
            for ( int i1 = 0; i1 <= m; i1++ )
            {
                for ( int j1 = 0; j1 <= n; j1++ )
                {
                    if ( i1 == 0 || j1 == 0 )
                        L[ i1, j1 ] = 0;
                    else if ( x[ i1 - 1 ] == y[ j1 - 1 ] )
                        L[ i1, j1 ] = L[ i1 - 1, j1 - 1 ] + 1;
                    else
                        L[ i1, j1 ] = Math.Max( L[ i1 - 1, j1 ], L[ i1, j1 - 1 ] );
                }
            }

            // Following code is used to print LCS 
            int index = L[ m, n ];
            int temp  = index;

            // Create a character array to store the lcs string 
            var lcs = new string[ index + 1 ];
                lcs[ index ] = string.Empty; // Set the terminating character 

            // Start from the right-most-bottom-most corner and 
            // one by one store characters in lcs[] 
            int i = m, j = n;
            while ( i > 0 && j > 0 )
            {
                // If current character in X[] and Y are same, then 
                // current character is part of LCS 
                if ( x[ i - 1 ] == y[ j - 1 ] )
                {
                    // Put current character in result 
                    lcs[ index - 1 ] = x[ i - 1 ];

                    // reduce values of i, j and index 
                    i--;
                    j--;
                    index--;
                }

                // If not same, then find the larger of two and 
                // go in the direction of larger value 
                else if ( L[ i - 1, j ] > L[ i, j - 1 ] )
                    i--;
                else
                    j--;
            }

            var al = new List< string >( temp );
            for ( var k = 0; k <= temp; k++ )
            {
                if ( !lcs[ k ].IsNullOrEmpty() )
                {
                    al.Add( lcs[ k ] );
                }
            }
            return al;
        }
    }
}
