using System.Diagnostics;

using M  = System.Runtime.CompilerServices.MethodImplAttribute;
using O  = System.Runtime.CompilerServices.MethodImplOptions;
using TP = System.Runtime.TargetedPatchingOptOutAttribute;

namespace System.Collections.Generic
{
    /// <summary>
    /// 
    /// </summary>
    [DebuggerTypeProxy( typeof(ICollectionDebugView<>) ), DebuggerDisplay("Count = {Count}"), Serializable]
    public sealed class DirectAccessList< T >: IList< T >, ICollection< T >, IEnumerable< T >
    {
        private const string REASON = "Performance critical to inline this type of method across NGen image boundaries";
        private const int MAX_CAPACITY_THRESHOLD = 0x7FFFFFFF /*int.MaxValue*/ - 0x400 * 0x400 /*1MB*/; /* => 2146435071 == 0x7fefffff*/
        //---private const int DEFAULT_CAPACITY = 4;
        private static readonly T[] EMPTY_ARRAY = new T[ 0 ];

        public   T[] _Items;
        internal int _Size;

		[M(O.AggressiveInlining), TP(REASON)] public DirectAccessList() => _Items = EMPTY_ARRAY;
		[M(O.AggressiveInlining), TP(REASON)] public DirectAccessList( int capacity ) => _Items = (capacity <= 0) ? EMPTY_ARRAY : new T[ capacity ];
        public DirectAccessList( IEnumerable< T > seq )
		{
			if ( seq == null ) throw (new ArgumentNullException( nameof(seq) ));

            if ( seq is ICollection< T > coll )
            {
                if ( coll.Count == 0 )
                {
                    _Items = EMPTY_ARRAY;
                }
                else
                {
                    _Items = new T[ coll.Count ];
                    coll.CopyTo( _Items, 0 );
                    _Size = _Items.Length;
                }
            }
            else
            {
                _Size  = 0;
                _Items = EMPTY_ARRAY;
                using ( var enumerator = seq.GetEnumerator() )
                {
                    while ( enumerator.MoveNext() )
                    {
                        Add( enumerator.Current );
                    }
                }
            }
        }

        public int Capacity
        {
            [M(O.AggressiveInlining), TP(REASON)] get => _Items.Length;
            [M(O.AggressiveInlining)] set
            {
                if ( value != _Items.Length )
                {
                    if ( 0 < value )
                    {
                        var array = new T[ value ];
                        if ( 0 < _Size )
                        {
                            Array.Copy( _Items, 0, array, 0, _Size );
                        }
                        _Items = array;
                    }
                    else
                    {
                        _Items = EMPTY_ARRAY;
                    }
                }
            }
        }
        public void Insert2Head( T item )
        {
            if ( _Size == _Items.Length )
            {
                EnsureCapacity( _Size + 1 );
            }
            if ( 0 < _Size )
            {
                Array.Copy( _Items, 0, _Items, 1, _Size );
            }
            _Items[ 0 ] = item;
            _Size++;
        }

        [M(O.AggressiveInlining)] private void EnsureCapacity( int min )
        {
            if ( _Items.Length < min )
            {
                var n = (_Items.Length == 0) ? 4 : (_Items.Length * 2);
                if ( n > MAX_CAPACITY_THRESHOLD )
                {
                    n = MAX_CAPACITY_THRESHOLD;
                }
                if ( n < min )
                {
                    n = min;
                }
                Capacity = n;
            }
        }

        #region [.IList< T >.]
        public T this[ int index ]
        {
            [M(O.AggressiveInlining), TP(REASON)] get => _Items[ index ];
            [M(O.AggressiveInlining), TP(REASON)] set => _Items[ index ] = value;
        }

        public void Insert( int index, T item )
        {
            if ( _Size == _Items.Length )
            {
                EnsureCapacity( _Size + 1 );
            }
            if ( index < _Size )
            {
                Array.Copy( _Items, index, _Items, index + 1, _Size - index );
            }
            _Items[ index ] = item;
            _Size++;
        }

        public int IndexOf( T item ) => throw (new NotImplementedException());
        public void RemoveAt( int index ) => throw (new NotImplementedException());
        #endregion

        #region [.ICollection< T >.]
        [M(O.AggressiveInlining)] public void Add( T item )
        {
            if ( _Size == _Items.Length )
            {
                EnsureCapacity( _Size + 1 );
            }
            _Items[ _Size++ ] = item;
        }
        [M(O.AggressiveInlining)] public void AddByRef( in T item )
        {
            if ( _Size == _Items.Length )
            {
                EnsureCapacity( _Size + 1 );
            }
            _Items[ _Size++ ] = item;
        }
        [M(O.AggressiveInlining)] public void Clear()
        {
            if ( 0 < _Size )
            {
                Array.Clear( _Items, 0, _Size );
                _Size = 0;
            }
        }

        public int Count { [M(O.AggressiveInlining), TP(REASON)] get => _Size; }

        public bool IsReadOnly { [M(O.AggressiveInlining), TP(REASON)] get => false; }
        public bool Remove( T item ) => throw (new NotImplementedException());
        public bool Contains( T item ) => throw (new NotImplementedException());
        public void CopyTo( T[] array, int arrayIndex )
        {
            foreach ( var t in this )
            {
                array[ arrayIndex++ ] = t;
            }
        }
        #endregion

        #region [.IEnumerable< T >.]
        public IEnumerator< T > GetEnumerator()
        {
            for ( var i = 0; i < _Size; i++ )
            {
                yield return (_Items[ i ]);
            }
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        #endregion
    }
}
