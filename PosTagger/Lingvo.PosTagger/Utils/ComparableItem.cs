using System;
using System.Collections.Generic;

namespace Lingvo.PosTagger.Utils
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class ComparableItem< T >
    {
        public ComparableItem( float score, T value )
        {
            Score = score;
            Value = value;
        }

        public float Score { get; }
        public T     Value { get; }
    }
    /// <summary>
    /// 
    /// </summary>
    internal sealed class ComparableItemComparer< T > : IComparer< ComparableItem< T > >
    {
        public static ComparableItemComparer< T > Asc { get; } = new ComparableItemComparer< T >( ascending: true );
        public static ComparableItemComparer< T > Desc { get; } = new ComparableItemComparer< T >( ascending: false );

        private bool _Ascending;
        private ComparableItemComparer( bool ascending ) => _Ascending = ascending;
        public int Compare( ComparableItem< T > x, ComparableItem< T > y )
        {
            var sign = Math.Sign( x.Score - y.Score );
            if ( !_Ascending )
            {
                sign = -sign;
            }
            return (sign);
        }
    }
}
