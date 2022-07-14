using System.Diagnostics;

using M  = System.Runtime.CompilerServices.MethodImplAttribute;
using O  = System.Runtime.CompilerServices.MethodImplOptions;
using TP = System.Runtime.TargetedPatchingOptOutAttribute;

namespace System.Collections.Generic
{
    /// <summary>
    /// 
    /// </summary>
    [DebuggerTypeProxy(typeof(ICollectionDebugView<>)), DebuggerDisplay("Count = {Count}")]
    public sealed class SetByRef< T > : ICollection< T >, IReadOnlyCollection< T > where T : struct
    {
        private const string REASON = "Performance critical to inline this type of method across NGen image boundaries";

        /// <summary>
        /// 
        /// </summary>
        private struct Slot
        {
            internal int HashCode;
            internal int Next;
            internal T   Value;
        }

        private const int DEFAULT_CAPACITY = 7;

        private int[]  _Buckets;
        private Slot[] _Slots;
        private int    _Count;
        private int    _FreeList;
        private IEqualityComparerByRef< T > _Comparer;

        public int Count { [M(O.AggressiveInlining)] get => _Count; }

        #region [.ctor().]
        [M(O.AggressiveInlining), TP(REASON)] public SetByRef( IEqualityComparerByRef< T > comparer, int capacity = DEFAULT_CAPACITY )
        {
            var capacityPrime = PrimeHelper.GetPrime( capacity );

            _Comparer = comparer ?? throw (new ArgumentNullException( nameof(comparer) ));
            _Buckets  = new int [ capacityPrime ];
            _Slots    = new Slot[ capacityPrime ];
            _FreeList = -1;
        }
        [M(O.AggressiveInlining), TP(REASON)] public SetByRef( IEqualityComparerByRef< T > comparer, IList< T > lst ) : this( comparer, lst, lst.Count ) { }
        [M(O.AggressiveInlining), TP(REASON)] public SetByRef( IEqualityComparerByRef< T > comparer, IEnumerable< T > seq, int? capacity = null ) 
            : this( comparer, capacity.GetValueOrDefault( DEFAULT_CAPACITY ) )
        {
            if ( seq != null )
            {
                foreach ( var t in seq )
                {
                    this.Add( in t );
                }
            }
        }

        [M(O.AggressiveInlining), TP(REASON)] private SetByRef() { }
        [M(O.AggressiveInlining), TP(REASON)] public static SetByRef< T > CreateWithCloserCapacity( IEqualityComparerByRef< T > comparer, int capacity )
        {
            if ( comparer == null ) throw (new ArgumentNullException( nameof(comparer) ));
            var capacityPrime = PrimeHelper.GetPrimeCloser( capacity );

            var set = new SetByRef< T >()
            {
                _Comparer = comparer,
                _Buckets  = new int [ capacityPrime ],
                _Slots    = new Slot[ capacityPrime ],
                _FreeList = -1,
            };
            return (set);
        }
        #endregion

        [M(O.AggressiveInlining)] public bool Add( in T value )
        {
            #region [.try find exists.]
            var hash   = InternalGetHashCode( in value );
            var bucket = hash % _Buckets.Length;
            for ( var i = _Buckets[ bucket ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( in slot.Value, in value ) )
                {
                    return (false);
                }
                i = slot.Next;
            }
            #endregion

            #region [.add new.]
            int index;
            if ( 0 <= _FreeList )
            {
                index = _FreeList;
                ref readonly var slot = ref _Slots[ index ];
                _FreeList = slot.Next;
            }
            else
            {
                if ( _Count == _Slots.Length )
                {
                    Resize();
                    bucket = hash % _Buckets.Length;
                }
                index = _Count;                
            }
            _Slots[ index ] = new Slot() 
            {
                HashCode = hash,
                Value    = value,
                Next     = _Buckets[ bucket ] - 1,
            };
            _Buckets[ bucket ] = index + 1;
            _Count++;
            return (true);
            #endregion            
        }
        [M(O.AggressiveInlining), TP(REASON)] public bool Contains( in T value )
        {
            var hash = InternalGetHashCode( in value );
            for ( var i = _Buckets[ hash % _Buckets.Length ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( in slot.Value, in value ) )
                {
                    return (true);
                }
                i = slot.Next;
            }
            return (false);
        }
        [M(O.AggressiveInlining)] public bool TryGetValue( in T value, out T existsValue )
        {
            var hash = InternalGetHashCode( in value );
            for ( var i = _Buckets[ hash % _Buckets.Length ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( in slot.Value, in value ) )
                {
                    existsValue = slot.Value;
                    return (true);
                }
                i = slot.Next;
            }
            existsValue = default;
            return (false);
        }

        public T First { [M(O.AggressiveInlining)] get => TryGetByRawIndex( 0 ); }
        [M(O.AggressiveInlining)] public T TryGetByRawIndex( int rawIndex )
        {
            for ( ; rawIndex < _Count; rawIndex++ )
            {                
                ref readonly var slot = ref _Slots[ rawIndex ];
                if ( 0 <= slot.HashCode )
                {
                    return (slot.Value);
                }
            }
            return (default);
        }

        public bool Remove( in T value )
        {
            var hash   = InternalGetHashCode( in value );
            var bucket = hash % _Buckets.Length;
            var last   = -1;
            for ( var i = _Buckets[ bucket ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( in slot.Value, in value ) )
                {
                    if ( last < 0 )
                    {
                        _Buckets[ bucket ] = slot.Next + 1; // _Slots[ i ].next + 1;
                    }
                    else
                    {
                        _Slots[ last ].Next = slot.Next; // _Slots[ i ].next;
                    }
                    _Slots[ i ] = new Slot()
                    {
                        HashCode = -1,
                        Next     = _FreeList,
                        //Value    = default,
                    };
                    _Count--;
                    _FreeList = (_Count == 0) ? -1 : i;
                    return (true);
                }
                last = i;
                i = slot.Next;
            }
            return (false);
        }

        [M(O.AggressiveInlining)] private void Resize()
        {
            var newSize    = PrimeHelper.ExpandPrime4Size( _Count ); // checked( _Count * 2 + 1 );
            var newBuckets = new int[ newSize ];
            var newSlots   = new Slot[ newSize ];
            Array.Copy( _Slots, 0, newSlots, 0, _Count );
            for ( int i = 0; i < _Count; i++ )
            {
                ref var slot = ref newSlots[ i ];
                var bucket = slot.HashCode % newSize;
                slot.Next = newBuckets[ bucket ] - 1;
                newBuckets[ bucket ] = i + 1;
            }
            _Buckets = newBuckets;
            _Slots   = newSlots;
        }
        [M(O.AggressiveInlining)] private int InternalGetHashCode( in T value ) => (_Comparer.GetHashCode( in value ) & 0x7FFFFFFF);

        #region [.ICollection< T >.]
        int ICollection< T >.Count => _Count;
        bool ICollection< T >.IsReadOnly => false;
        void ICollection< T >.Add( T item ) => Add( in item );
        void ICollection< T >.Clear()
        {
            //var capacity = _Slots.Length;
            //_Buckets  = new int [ capacity ];
            //_Slots    = new Slot[ capacity ];
            Array.Clear( _Buckets, 0, _Buckets.Length );
            Array.Clear( _Slots  , 0, _Slots  .Length );
            _FreeList = -1;
            _Count    = 0;
        }
        bool ICollection< T >.Contains( T item ) => Contains( in item );
        void ICollection< T >.CopyTo( T[] array, int arrayIndex )
        {
            foreach ( var item in this )
            {
                array[ arrayIndex++ ] = item;
            }
        }
        bool ICollection< T >.Remove( T item ) => Remove( in item );
        #endregion

        #region [.IEnumerator< T >.]
        public IEnumerator< T > GetEnumerator() => new Enumerator( this );
        IEnumerator IEnumerable.GetEnumerator() => new Enumerator( this );

        /// <summary>
        /// 
        /// </summary>
        public struct Enumerator : IEnumerator< T >
        {
            private SetByRef< T > _Set;
            private int           _Index;

            internal Enumerator( SetByRef< T > set )
            {
                _Set    = set;
                _Index  = 0;
                Current = default;
            }
            public void Dispose() { }

            public bool MoveNext()
            {
                // Use unsigned comparison since we set index to set.count+1 when the enumeration ends.
                // set.count+1 could be negative if dictionary.count is Int32.MaxValue
                while ( (uint) _Index < (uint) _Set._Count )
                {
                    ref readonly var slot = ref _Set._Slots[ _Index ];
                    if ( 0 <= slot.HashCode )
                    {
                        Current = slot.Value;
                        _Index++;
                        return (true);
                    }
                    _Index++;
                }

                _Index  = _Set._Count + 1;
                Current = default;
                return (false);
            }
            public T Current { [M(O.AggressiveInlining)] get; [M(O.AggressiveInlining)] private set; }
            public void Reset()
            {
                _Index  = 0;
                Current = default;
            }

            object IEnumerator.Current => Current;
            void IEnumerator.Reset() => Reset();
        }
        #endregion
    }
}
