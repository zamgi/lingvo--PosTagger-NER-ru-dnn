using System.Collections.Generic;
using System.Linq;

using Lingvo.PosTagger.Network;

namespace Lingvo.PosTagger.Utils
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class BeamSearchStatus
    {
        public BeamSearchStatus()
        {
            OutputIds = new List<int>();
            HTs = new List<WeightTensor>();
            CTs = new List<WeightTensor>();

            Score = 1.0f;
        }

        public List<int> OutputIds { get; }
        public float Score { get; }

        public List<WeightTensor> HTs { get; }
        public List<WeightTensor> CTs { get; }
    }
    /// <summary>
    /// 
    /// </summary>
    public static class BeamSearch
    {
        public static List< BeamSearchStatus > GetTopNBSS( List< BeamSearchStatus > bssList, int topN )
        {
            var q = new FixedSizePriorityQueue< ComparableItem< BeamSearchStatus > >( topN, ComparableItemComparer< BeamSearchStatus >.Desc );
            for ( int i = 0; i < bssList.Count; i++ )
            {
                q.Enqueue( new ComparableItem< BeamSearchStatus >( bssList[ i ].Score, bssList[ i ] ) );
            }
            return (q.Select( x => x.Value ).ToList( q.Count ));
        }
    }
}
