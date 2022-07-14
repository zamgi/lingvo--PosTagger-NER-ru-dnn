using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace System.Collections.Generic
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class IntervalList< T > : IEnumerable< (IntervalList< T >.Interval key, T value) >
	{
        /// <summary>
        /// 
        /// </summary>
        public struct Interval
        {
            public int startIndex;
            public int length;
            public int endIndex() => startIndex + length;
            public override string ToString() => $"{startIndex}:{length}";

            public static implicit operator Interval( in (int startIndex, int length) t ) => new Interval() { startIndex = t.startIndex, length = t.length };
        }

        private const int MAX_CAPACITY_THRESHOLD = 0x7FFFFFFF /*int.MaxValue*/ - 0x400 * 0x400 /*1MB*/; /* => 2146435071 == 0x7fefffff*/
        private static readonly (Interval key, T value)[] EMPTY_ARRAY = new (Interval key, T value)[ 0 ];

        private (Interval key, T value)[] _Array;
		private int _Size;
        
        public IntervalList( int capacity = 0 ) => _Array = (capacity == 0) ? EMPTY_ARRAY : new (Interval key, T value)[ capacity ];

        public void Add( in (int startIndex, int length) t, T value ) => Add( (Interval) t, value );
        public void Add( in Interval key, T value )
        {
            var index = InternalBinarySearch( key );
            if ( 0 <= index )
            {
                throw (new ArgumentException( $"Key already exists: {key}" ));
            }
            Insert( ~index, key, value );
        }

        public bool TryAdd( in (int startIndex, int length) t, T value ) => TryAdd( (Interval) t, value );
        public bool TryAdd( in Interval key, T value )
        {
            var index = InternalBinarySearch( key );
            if ( 0 <= index )
            {
                return (false);
            }
            Insert( ~index, key, value );
            return (true);
        }
        public bool TryGetValue( in (int startIndex, int length) t, out T exists ) => TryGetValue( (Interval) t, out exists );
        public bool TryGetValue( in Interval key, out T exists )
        {
            var index = InternalBinarySearch__4Search( key );
            if ( 0 <= index )
            {
                ref readonly var t = ref _Array[ index ];
                exists = t.value;
                return (true);
            }
            exists = default;
            return (false); 
        }
        private void RemoveAt( int index )
		{
            //if ( index < 0 || _Size <= index ) throw (new ArgumentOutOfRangeException( nameof(index) ));
			
            _Size--;
            if ( index < _Size )
			{
                Array.Copy( _Array, index + 1, _Array, index, _Size - index );
			}
			_Array[ _Size ] = default;
		}
        public bool Remove( in (int startIndex, int length) t ) => Remove( (Interval) t );
        public bool Remove( in Interval key )
		{
            var index = InternalBinarySearch__4Search( key );
            if ( 0 <= index )
            {
                RemoveAt( index );
            }
            return (0 <= index);
		}
        public void Clear()
        {
            _Size = 0;
            Array.Clear( _Array, 0, _Array.Length );
        }

        [M(O.AggressiveInlining)] private void Insert( int index, in Interval key, T value )
		{
            if ( _Size == _Array.Length )
            {
                EnsureCapacity( _Size + 1 );
            }
            if ( index < _Size )
            {
                Array.Copy( _Array, index, _Array, index + 1, _Size - index );
            }
            _Array[ index ] = (key, value);
			_Size++;
		}
        private void EnsureCapacity( int min )
		{
            int capacity;
            switch ( _Array.Length )
            {
                case 0:  capacity = 1; break;
                case 1:  capacity = 16; break;
                default: capacity = _Array.Length * 2; break;
            }
            if ( MAX_CAPACITY_THRESHOLD < capacity )
            {
                capacity = MAX_CAPACITY_THRESHOLD;
            }
            if ( capacity < min )
            {
                capacity = min;
            }
			Capacity = capacity;
		}

		public int Capacity
		{
			get => _Array.Length;
			private set
			{
                if ( value != _Array.Length )
				{
                    if ( 0 < value )
					{
                        var destinationArray = new (Interval key, T value)[ value ];
                        if ( 0 < _Size )
						{
                            Array.Copy( _Array, 0, destinationArray, 0, _Size );
						}
                        _Array = destinationArray;
					}
                    else
                    {
                        _Array = EMPTY_ARRAY;
                    }
				}
			}
		}
		public int Count { [M(O.AggressiveInlining)] get => _Size; }

        [M(O.AggressiveInlining)] private bool IsIntersect( in Interval x, in Interval y ) => (x.startIndex <= y.startIndex && y.startIndex <= x.endIndex()) ||
                                                                                              (y.startIndex <= x.startIndex && x.startIndex <= y.endIndex());
        [M(O.AggressiveInlining)] private int InternalBinarySearch( in Interval searchKey )
        {
            var i = 0;
            for ( var endIndex = _Size - 1; i <= endIndex; )
            {
                var middleIndex = i + ((endIndex - i) >> 1);
                ref readonly var t = ref _Array[ middleIndex ];

                //---var d = string.CompareOrdinal( t.key, searchKey );
                //if ( d == 0 )
                //{
                //    return (middleIndex);
                //}
                if  ( IsIntersect( t.key, searchKey ) )
                {
                    return (middleIndex);
                }
                //var d = t.key.startIndex.CompareTo( searchKey.startIndex );
                //if ( d < 0 )
                if ( t.key.startIndex < searchKey.startIndex )
                {
                    i = middleIndex + 1;
                }
                else
                {
                    endIndex = middleIndex - 1;
                }
            }
            return (~i);
        }
        [M(O.AggressiveInlining)] private int InternalBinarySearch__4Search( in Interval searchKey )
        {
            var i = 0;
            for ( var endIndex = _Size - 1; i <= endIndex; )
            {
                var middleIndex = i + ((endIndex - i) >> 1);
                ref readonly var t = ref _Array[ middleIndex ];

                if  ( IsIntersect( t.key, searchKey ) )
                {
                    for ( i = middleIndex - 1; 0 <= i; i-- )
                    {
                        t = ref _Array[ i ];
                        if ( !IsIntersect( t.key, searchKey ) )
                        {
                            break;
                        }
                        middleIndex = i;
                    }
                    return (middleIndex);
                }
                if ( t.key.startIndex < searchKey.startIndex )
                {
                    i = middleIndex + 1;
                }
                else
                {
                    endIndex = middleIndex - 1;
                }
            }
            return (~i);
        }

        #region [.IEnumerable<>.]
        public IEnumerable< T > GetValues()
        {
            for ( int i = 0; i < _Size; i++ )
            {
                yield return (_Array[ i ].value);
            }
        }
        public IEnumerator< (Interval key, T value) > GetEnumerator()
        {
            for ( int i = 0; i < _Size; i++ )
            {
                yield return (_Array[ i ]);
            }
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        #endregion

#if DEBUG
        public override string ToString() => $"Count: {Count}";
#endif
    }
}
