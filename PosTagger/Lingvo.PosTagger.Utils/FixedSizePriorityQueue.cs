using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lingvo.PosTagger.Utils
{
    /// <typeparam name="T">T must be comparable.</typeparam>
    public class FixedSizePriorityQueue< T > : IEnumerable< T >
    {
        #region [.fields.]
        protected IComparer< T > _Comparer;

        /// <summary>
        /// The downward facing heap.
        /// </summary>
        protected T[] _Bottom;
        /// <summary>
        /// Upward facing heap.
        /// </summary>
        protected T[] _Top;
        /// <summary>
        /// Total number of elements in the heap.
        /// </summary>
        protected int _Count;
        /// <summary>
        /// Capacity of the heap.
        /// </summary>
        protected int _Capacity;
        /// <summary>
        /// Number of nodes in the bottom heap.
        /// </summary>
        protected int _BottomSize;
        /// <summary>
        /// Number of nodes in the top heap.
        /// </summary>
        protected int _TopSize;
        #endregion

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="capacity">Fix the heap size at this capacity.</param>
        public FixedSizePriorityQueue( int capacity )
        {
            if ( capacity < 1 ) throw (new ArgumentNullException( "priority queue capacity must be at least one!" ));

            _Count      = 0;
            _Capacity   = capacity;
            _BottomSize = _TopSize = 0;

            var bottomCapacity = Math.Max( capacity / 2, 1 );
            var topCapacity    = Math.Max( capacity - bottomCapacity, 1 );

            _Top    = new T[ topCapacity ];
            _Bottom = new T[ bottomCapacity ];
        }
        public FixedSizePriorityQueue( int capacity, IComparer< T > comparer ): this( capacity ) => _Comparer = comparer;

        #region Basic prio-queue operations.
        /// <summary>
        /// Get the number of elements currently in the queue.
        /// </summary>
        public int Count => _Count;

        /// <summary>
        /// Get the number of elements that could possibly be held in the queue
        /// </summary>
        public int Capacity => _Capacity;

        public void Clear()
        {
            _Count = 0;
            _Bottom.Initialize();
            _Top.Initialize();
            _TopSize = _BottomSize = 0;
        }

        public bool Enqueue( T t )
        {
            // first, are we already at capacity?
            if ( _Capacity == _Count )
            {
                if ( _Capacity == 1 )
                {
                    //We only have a single item in the queue and it's kept in m_rgtTop.
                    if ( !Better( t, _Top[ 0 ], true ) )
                    {
                        return (false);
                    }
                    else
                    {
                        _Top[ 0 ] = t;
                        return (true);
                    }
                }

                // then, are we better than the bottom?
                if ( !Better( t, _Bottom[ 0 ], true ) )
                {
                    // nope, bail.
                    return (false);
                }

                // yep, put in place...
                _Bottom[ 0 ] = t;
                // first heapfiy the bottom half; get back the
                // index where it ended up.
                int updated = DownHeapify( _Bottom, 0, _BottomSize, false );

                // are we not at the boundary?  Then we're done.
                if ( !SemiLeaf( updated, _BottomSize ) ) return (true);

                // at the boundary: check if we need to update.
                int top = CheckBoundaryUpwards( updated );

                // boundary is okay?  bail.
                if ( top == -1 ) return (true);

                // ...and fix the top heap property.
                UpHeapify( _Top, top, true );

                return (true);
            }

            // we have space to insert.
            ++_Count;
            // need to maintain the invariant that either size(bottom) == size(top),
            // or size(bottom) + 1 == size(top).
            if ( _BottomSize < _TopSize )
            {
                Debug.Assert( _BottomSize + 1 == _TopSize );
                // bottom is smaller: put it there.
                int pos = _BottomSize++;
                _Bottom[ pos ] = t;

                // see if it should really end up in the top heap...
                int up = CheckBoundaryUpwards( pos );
                if ( up == -1 )
                {
                    // no -- fix the bottom yep.
                    UpHeapify( _Bottom, pos, false );
                }
                else
                {
                    // yes -- fix the top heap.
                    UpHeapify( _Top, up, true );
                }
                return (true);
            }
            else
            {
                Debug.Assert( _BottomSize == _TopSize );
                // put it in the top.
                int pos = _TopSize++;
                _Top[ pos ] = t;

                // see if it should really end up in the bottom.
                int bottom = CheckBoundaryDownwards( pos );
                if ( bottom == -1 )
                {
                    // no -- fix the top heap.
                    UpHeapify( _Top, pos, true );
                }
                else
                {
                    // yes -- fix the bottotm.
                    UpHeapify( _Bottom, bottom, false );
                }
                return (true);
            }
        }
        #endregion

        #region Heap navigators
        private int Parent( int i ) => (i - 1) / 2;
        private int Left( int i ) => 2 * i + 1;
        private int Right( int i ) => 2 * i + 2;
        private bool SemiLeaf( int i, int iSize ) => (Right( i ) >= iSize);
        //private bool IsLeft( int i ) => i % 2 == 1;
        //private bool IsRight( int i ) => i % 2 == 0;
        //private bool Leaf( int i, int iSize ) => (Left( i ) >= iSize);

        private int BottomNode( int i )
        {
            // first see if we have a direct correspondence.
            if ( i < _BottomSize ) return (i);

            // no parallel -- must be that one extra element
            // in the target heap.  instead point at the parent
            // if a left node, or left sibling if a right node.
            Debug.Assert( i <= _BottomSize );
            if ( i % 2 == 1 ) return (Parent( i ));
            return (i - 1);
        }

        private int TopNode1( int i )
        {
            if ( Left( i ) >= _BottomSize && Left( i ) < _TopSize ) return (Left( i ));
            // top is always >= bottom in size,
            // so this element is guaranteed to exist.
            return (i);
        }

        private int TopNode2( int i )
        {
            if ( i == _BottomSize - 1 &&
                1 == (i % 2) &&
                _TopSize > _BottomSize )
            {
                return (i + 1);
            }
            if ( Left( i ) >= _BottomSize && Left( i ) < _TopSize ) return (Left( i ));

            return (i);
        }
        #endregion

        #region Heap invariant maintenance
        private int UpHeapify( T[] rgt, int i, bool isTop )
        {
            while ( i > 0 )
            {
                int par = Parent( i );
                if ( !Better( rgt[ i ], rgt[ par ], isTop ) ) return (i);
                Swap( rgt, i, rgt, par );
                i = par;
            }
            return (i);
        }

        private int DownHeapify( T[] rgt, int i, int size, bool isTop )
        {
            while ( true )
            {
                int left = Left( i ), right = Right( i );
                int largest = i;
                if ( left < size && Better( rgt[ left ], rgt[ largest ], isTop ) ) largest = left;
                if ( right < size && Better( rgt[ right ], rgt[ largest ], isTop ) ) largest = right;
                if ( largest == i ) return i;

                Swap( rgt, i, rgt, largest );
                i = largest;
            }
        }

        private int CheckBoundaryUpwards( int bottomPos )
        {
            int top1 = TopNode1( bottomPos );
            int top2 = TopNode2( bottomPos );
            int better = -1;
            if ( Better( _Bottom[ bottomPos ], _Top[ top1 ], true ) )
            {
                better = top1;
            }
            if ( Better( _Bottom[ bottomPos ], _Top[ top2 ], true ) &&
                (better == -1 || Better( _Top[ top1 ], _Top[ top2 ], true )) )
            {
                better = top2;
            }
            if ( better == -1 )
            {
                return -1;
            }

            // boundary is not okay? move this guy across...
            Swap( _Top, better, _Bottom, bottomPos );

            return better;
        }

        private int CheckBoundaryDownwards( int topPos )
        {
            // compare to the bottom guy in the corresponding posn.
            int bottomPos = BottomNode( topPos );
            if ( bottomPos == -1 )
            {
                return -1;
            }
            if ( (bottomPos >= _BottomSize) || !Better( _Bottom[ bottomPos ], _Top[ topPos ], true ) )
            {
                return -1;
            }

            Swap( _Top, topPos, _Bottom, bottomPos );

            return (bottomPos);
        }
        protected bool Better( T t1, T t2, bool isTop )
        {
            var i = (_Comparer == null) ? ((IComparable< T >) t1).CompareTo( t2 ) : _Comparer.Compare( t1, t2 );
            //int i = comparer.Compare(t1, t2);

            return (!isTop ? i < 0 : i > 0);
        }

        private static void Swap( T[] rgt1, int i1, T[] rgt2, int i2 )
        {
            T t = rgt1[ i1 ];
            rgt1[ i1 ] = rgt2[ i2 ];
            rgt2[ i2 ] = t;
        }
        #endregion

        #region IEnumerable< T >
        public IEnumerator< T > GetEnumerator()
        {
            for ( var i = 0; i < _TopSize; ++i )
            {
                yield return _Top[ i ];
            }
            for ( var i = _BottomSize - 1; i >= 0; --i )
            {
                yield return _Bottom[ i ];
            }
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        #endregion
    }
}

