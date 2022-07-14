using System.Diagnostics;

using M  = System.Runtime.CompilerServices.MethodImplAttribute;
using O  = System.Runtime.CompilerServices.MethodImplOptions;
using TP = System.Runtime.TargetedPatchingOptOutAttribute;

namespace System.Collections.Generic
{
    /// <summary>
    /// 
    /// </summary>
    [DebuggerTypeProxy( typeof(IMapCollectionDebugView<,>) ), DebuggerDisplay("Count = {Count}")]
    public sealed class Map< K, T > : ICollection< KeyValuePair< K, T > >, IReadOnlyCollection< KeyValuePair< K, T > >
    {
        private const string REASON = "Performance critical to inline this type of method across NGen image boundaries";

        /// <summary>
        /// 
        /// </summary>
        private struct Slot
        {
            internal int HashCode;
            internal int Next;
            internal K   Key;
            internal T   Value;
        }

        private const int DEFAULT_CAPACITY = 7;

        private int[]  _Buckets;
        private Slot[] _Slots;
        private int    _Count;
        private int    _FreeList;
        private IEqualityComparer< K > _Comparer;

        public int Count { [M(O.AggressiveInlining)] get => _Count; }
        public IEqualityComparer< K > Comparer { [M(O.AggressiveInlining)] get => _Comparer; }

        #region [.ctor().]
        [M(O.AggressiveInlining), TP(REASON)] public Map( IEqualityComparer< K > comparer ) : this( DEFAULT_CAPACITY, comparer ) { }
        [M(O.AggressiveInlining), TP(REASON)] public Map( int capacity = DEFAULT_CAPACITY, IEqualityComparer< K > comparer = null )
        {
            var capacityPrime = PrimeHelper.GetPrime( capacity );

            _Comparer = comparer ?? EqualityComparer< K >.Default;
            _Buckets  = new int [ capacityPrime ];
            _Slots    = new Slot[ capacityPrime ];
            _FreeList = -1;
        }

        [M(O.AggressiveInlining), TP(REASON)] private Map() { }
        [M(O.AggressiveInlining), TP(REASON)] public static Map< K, T > CreateWithCloserCapacity( int capacity, IEqualityComparer< K > comparer = null )
        {
            var capacityPrime = PrimeHelper.GetPrimeCloser( capacity );

            var map = new Map< K, T >()
            {
                _Comparer = comparer ?? EqualityComparer< K >.Default,
                _Buckets  = new int [ capacityPrime ],
                _Slots    = new Slot[ capacityPrime ],
                _FreeList = -1,
            };
            return (map);
        }
        #endregion

        [M(O.AggressiveInlining)] public bool Add( K key, T value )
        {
            #region [.try find exists.]
            var hash   = InternalGetHashCode( key );
            var bucket = hash % _Buckets.Length;
            for ( var i = _Buckets[ bucket ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( slot.Key, key ) )
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
                Key      = key,
                Next     = _Buckets[ bucket ] - 1,
            };
            _Buckets[ bucket ] = index + 1;
            _Count++;
            return (true);
            #endregion            
        }
        [M(O.AggressiveInlining)] public bool TryAdd( K key, T value, out T existsValue )
        {
            #region [.try find exists.]
            var hash   = InternalGetHashCode( key );
            var bucket = hash % _Buckets.Length;
            for ( var i = _Buckets[ bucket ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( slot.Key, key ) )
                {
                    existsValue = slot.Value;
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
                Key      = key,
                Next     = _Buckets[ bucket ] - 1,
            };
            _Buckets[ bucket ] = index + 1;
            _Count++;
            existsValue = default;
            return (true);
            #endregion            
        }
        [M(O.AggressiveInlining)] public void AddOrUpdate( K key, T value )
        {
            #region [.try find exists.]
            var hash   = InternalGetHashCode( key );
            var bucket = hash % _Buckets.Length;
            for ( var i = _Buckets[ bucket ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( slot.Key, key ) )
                {
                    slot.Value = value;
                    return;
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
                Key      = key,
                Next     = _Buckets[ bucket ] - 1,
            };
            _Buckets[ bucket ] = index + 1;
            _Count++;
            #endregion
        }
        [M(O.AggressiveInlining)] public bool TryUpdate( K key, T value )
        {
            #region [.try find exists.]
            var hash   = InternalGetHashCode( key );
            var bucket = hash % _Buckets.Length;
            for ( var i = _Buckets[ bucket ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( slot.Key, key ) )
                {
                    slot.Value = value;
                    return (true);
                }
                i = slot.Next;
            }
            return (false);
            #endregion  
        }
        [M(O.AggressiveInlining), TP(REASON)] public bool ContainsKey( K key )
        {
            var hash = InternalGetHashCode( key );
            for ( var i = _Buckets[ hash % _Buckets.Length ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( slot.Key, key ) )
                {
                    return (true);
                }
                i = slot.Next;
            }
            return (false);
        }
        [M(O.AggressiveInlining)] public bool TryGetValue( K key, out T existsValue )
        {
            var hash = InternalGetHashCode( key );
            for ( var i = _Buckets[ hash % _Buckets.Length ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( slot.Key, key ) )
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

        public bool Remove( K key )
        {
            var hash   = InternalGetHashCode( key );
            var bucket = hash % _Buckets.Length;
            var last   = -1;
            for ( var i = _Buckets[ bucket ] - 1; 0 <= i; /*i = _Slots[ i ].next*/ )
            {
                ref readonly var slot = ref _Slots[ i ];
                if ( (slot.HashCode == hash) && _Comparer.Equals( slot.Key, key ) )
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
            var newBuckets = new int [ newSize ];
            var newSlots   = new Slot[ newSize ];
            Array.Copy( _Slots, 0, newSlots, 0, _Count );
            for ( var i = 0; i < _Count; i++ )
            {
                ref var slot = ref newSlots[ i ];
                var bucket = slot.HashCode % newSize;
                slot.Next = newBuckets[ bucket ] - 1;
                newBuckets[ bucket ] = i + 1;
            }
            _Buckets = newBuckets;
            _Slots   = newSlots;
        }
        [M(O.AggressiveInlining)] private int InternalGetHashCode( K key ) => (_Comparer.GetHashCode( key ) & 0x7FFFFFFF); //((key != null) ? (_Comparer.GetHashCode(value ) & 0x7FFFFFFF) : 0);

        #region [.ICollection< KeyValuePair< K, T > >.]
        int ICollection< KeyValuePair< K, T > >.Count => _Count;
        bool ICollection< KeyValuePair< K, T > >.IsReadOnly => false;
        void ICollection< KeyValuePair< K, T > >.Add( KeyValuePair< K, T > p ) => Add( p.Key, p.Value );
        void ICollection< KeyValuePair< K, T > >.Clear()
        {
            //var capacity = _Slots.Length;
            //_Buckets  = new int[ capacity ];
            //_Slots    = new Slot[ capacity ];
            Array.Clear( _Buckets, 0, _Buckets.Length );
            Array.Clear( _Slots  , 0, _Slots  .Length );
            _FreeList = -1;
            _Count    = 0;
        }
        bool ICollection< KeyValuePair< K, T > >.Contains( KeyValuePair< K, T > p ) => ContainsKey( p.Key );
        void ICollection< KeyValuePair< K, T > >.CopyTo( KeyValuePair< K, T >[] array, int arrayIndex )
        {
            foreach ( var p in this )
            {
                array[ arrayIndex++ ] = p;
            }
        }
        bool ICollection< KeyValuePair< K, T > >.Remove( KeyValuePair< K, T > p ) => Remove( p.Key );
        #endregion

        #region [.IEnumerator< KeyValuePair< K, T > >.]
        public IEnumerator< KeyValuePair< K, T > > GetEnumerator() => new Enumerator( this );
        IEnumerator IEnumerable.GetEnumerator() => new Enumerator( this );
        /*public IEnumerator< KeyValuePair< K, T > > GetEnumerator()
        {
            for ( var i = 0; i < _Count; i++ )
            {
                ref var slot = ref _Slots[ i ];
                if ( 0 <= slot.HashCode )
                {
                    yield return (new KeyValuePair< K, T >( slot.Key, slot.Value ));
                }
            }
        }*/

        /// <summary>
        /// 
        /// </summary>
        public struct Enumerator : IEnumerator< KeyValuePair< K, T > >
        {
            private Map< K, T > _Map;
            private int         _Index;

            [M(O.AggressiveInlining)] internal Enumerator( Map< K, T > map )
            {
                _Map    = map;
                _Index  = 0;
                Current = default;
            }
            public void Dispose() { }

            public bool MoveNext()
            {
                // Use unsigned comparison since we set index to set.count+1 when the enumeration ends.
                // set.count+1 could be negative if dictionary.count is Int32.MaxValue
                while ( (uint) _Index < (uint) _Map._Count )
                {
                    ref readonly var slot = ref _Map._Slots[ _Index ];
                    if ( 0 <= slot.HashCode )
                    {
                        Current = new KeyValuePair< K, T >( slot.Key, slot.Value );
                        _Index++;
                        return (true);
                    }
                    _Index++;
                }

                _Index  = _Map._Count + 1;
                Current = default;
                return (false);
            }
            public KeyValuePair< K, T > Current { [M(O.AggressiveInlining)] get; [M(O.AggressiveInlining)] private set; }
            public void Reset()
            {
                _Index  = 0;
                Current = default;
            }

            object IEnumerator.Current => Current;
            void IEnumerator.Reset() => Reset();
        }

        public IEnumerator< K > GetKeys() => new Enumerator4Keys( this );
        public ReadOnlyCollection4Values GetValues() => new ReadOnlyCollection4Values( this );

        /// <summary>
        /// 
        /// </summary>
        public struct Enumerator4Keys : IEnumerator< K >
        {
            private Map< K, T > _Map;
            private int         _Index;

            [M(O.AggressiveInlining)] internal Enumerator4Keys( Map< K, T > map )
            {
                _Map    = map;
                _Index  = 0;
                Current = default;
            }
            public void Dispose() { }

            public bool MoveNext()
            {
                // Use unsigned comparison since we set index to set.count+1 when the enumeration ends.
                // set.count+1 could be negative if dictionary.count is Int32.MaxValue
                while ( (uint) _Index < (uint) _Map._Count )
                {
                    ref readonly var slot = ref _Map._Slots[ _Index ];
                    if ( 0 <= slot.HashCode )
                    {
                        Current = slot.Key;
                        _Index++;
                        return (true);
                    }
                    _Index++;
                }

                _Index  = _Map._Count + 1;
                Current = default;
                return (false);
            }
            public K Current { [M(O.AggressiveInlining)] get; [M(O.AggressiveInlining)] private set; }
            public void Reset()
            {
                _Index  = 0;
                Current = default;
            }

            object IEnumerator.Current => Current;
            void IEnumerator.Reset() => Reset();
        }

        /// <summary>
        /// 
        /// </summary>
        public struct Enumerator4Values : IEnumerator< T >
        {
            private Map< K, T > _Map;
            private int         _Index;

            [M(O.AggressiveInlining)] internal Enumerator4Values( Map< K, T > map )
            {
                _Map    = map;
                _Index  = 0;
                Current = default;
            }
            public void Dispose() { }

            public bool MoveNext()
            {
                // Use unsigned comparison since we set index to set.count+1 when the enumeration ends.
                // set.count+1 could be negative if dictionary.count is Int32.MaxValue
                while ( (uint) _Index < (uint) _Map._Count )
                {
                    ref readonly var slot = ref _Map._Slots[ _Index ];
                    if ( 0 <= slot.HashCode )
                    {
                        Current = slot.Value;
                        _Index++;
                        return (true);
                    }
                    _Index++;
                }

                _Index  = _Map._Count + 1;
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
        /// <summary>
        /// 
        /// </summary>
        public struct ReadOnlyCollection4Values : IReadOnlyCollection< T >
        {
            private Map< K, T > _Map;
            [M(O.AggressiveInlining)] internal ReadOnlyCollection4Values( Map< K, T > map ) => _Map = map;

            public int Count => _Map.Count;
            //---[M(O.AggressiveInlining)] public bool HasValues() => ((_Map != null) && (0 <_Map.Count));

            public IEnumerator< T > GetEnumerator() => new Enumerator4Values( _Map );
            IEnumerator IEnumerable.GetEnumerator() => new Enumerator4Values( _Map );
        }
        #endregion
    }
}
