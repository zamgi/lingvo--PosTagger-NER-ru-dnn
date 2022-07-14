using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

using Lingvo.PosTagger.Tensors.Core;
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public class Tensor : IDisposable
    {
        private long[] _Sizes;
        private long[] _Strides;
        private readonly Storage _Storage;
        private readonly long _StorageOffset;
        private bool _IsDisposed;

        /// <summary>
        /// Construct a new tensor, using the given allocator to construct a storage. The new tensor
        /// will be contiguous in memory. The tensor's elements will not be initialized.
        /// </summary>
        public Tensor( IAllocator allocator, DType elementType, params long[] sizes ) : this( allocator, elementType, sizes, TensorDimensionHelpers.GetContiguousStride( sizes ) ) { }
        public Tensor( IAllocator allocator, DType elementType, long[] sizes, long[] strides )
        {
            _Sizes = sizes;
            _Strides = strides;
            _StorageOffset = 0;
            _Storage = allocator.Allocate( elementType, TensorDimensionHelpers.GetStorageSize( sizes, strides ) );
        }
        public Tensor( long[] sizes, long[] strides, Storage storage, long storageOffset )
        {
            _Sizes = sizes;
            _Strides = strides;
            _Storage = storage;
            _StorageOffset = storageOffset;

            _Storage.AddRef();
        }
        public void Dispose()
        {
            if ( !_IsDisposed )
            {
                _IsDisposed = true;
                _Storage.Release();
            }
            else
            {
                throw (new ObjectDisposedException( "Tensor" ));
            }
        }

        public override bool Equals( object obj )
        {
            if ( !(obj is Tensor o) )
            {
                return (false);
            }

            return (object.ReferenceEquals( _Storage, o._Storage ) &&
                    (_StorageOffset == o._StorageOffset) &&
                    TensorResultBuilder.ArrayEqual( _Sizes, o._Sizes ) &&
                    TensorResultBuilder.ArrayEqual( _Strides, o._Strides ));
        }
        public override int GetHashCode() => _Storage.GetHashCode() ^
                                             _StorageOffset.GetHashCode() ^
                                             _Sizes.Aggregate( 0, (acc, item) => acc ^ item.GetHashCode() ) ^
                                             _Strides.Aggregate( 0, (acc, item) => acc ^ item.GetHashCode() );

        public DType ElementType => _Storage.ElementType;
        public long[] Sizes => _Sizes;
        public long[] Strides => _Strides;
        public Storage Storage => _Storage;
        public long StorageOffset => _StorageOffset;
        public IAllocator Allocator => _Storage.Allocator;
        public int DimensionCount => _Sizes.Length;

        public long GetStorageSize() => TensorDimensionHelpers.GetStorageSize( _Sizes, _Strides );

        /// <summary>
        /// Returns a new Tensor object which points to the same storage as this,
        /// incrementing the refcount of the storage object.
        /// </summary>
        public Tensor CopyRef() => new Tensor( _Sizes, _Strides, _Storage, _StorageOffset );

        public bool IsOwnerExclusive() => _Storage.IsOwnerExclusive();
        public string Format() => TensorFormatting.Format( this );

        private long? _ElementCount;
        public long ElementCount()
        {
            if ( _ElementCount.HasValue )
            {
                return (_ElementCount.Value);
            }

            _ElementCount = TensorDimensionHelpers.ElementCount( _Sizes );
            return (_ElementCount.Value);
        }

        public bool IsContiguous()
        {
            long z = 1;
            for ( int d = _Sizes.Length - 1; d >= 0; d-- )
            {
                if ( _Sizes[ d ] != 1 )
                {
                    if ( _Strides[ d ] == z )
                    {
                        z *= _Sizes[ d ];
                    }
                    else
                    {
                        return (false);
                    }
                }
            }
            return (true);
        }

        [M(O.AggressiveInlining)] public bool IsSameSizeAs( Tensor other ) => TensorResultBuilder.ArrayEqual( _Sizes, other._Sizes );

        /// <summary>
        /// Note: this does not check whether indices are in range
        /// </summary>
        public float GetElementAsFloat( params long[] indices )
        {
            if ( indices.Length != DimensionCount ) throw (new ArgumentException( $"Number of indices must equal number of tensor dimensions. Tensor dim = '{DimensionCount}' and input indices length is '{indices.Length}'" ));

            for ( int i = 0; i < indices.Length; ++i )
            {
                if ( indices[ i ] < 0 || indices[ i ] >= Sizes[ i ] )
                {
                    throw (new ArgumentException( $"Index {i} with value {indices[ i ]} is out of range" ));
                }
            }

            long offset = 0;
            for ( int i = 0; i < indices.Length; ++i )
            {
                offset += indices[ i ] * _Strides[ i ];
            }
            return (_Storage.GetElementAsFloat( _StorageOffset + offset ));
        }

        public float[] GetElementsAsFloat( int length ) => _Storage.GetElementsAsFloat( _StorageOffset, length );

        /// <summary>
        /// Note: this does not check whether indices are in range
        /// </summary>
        public void SetElementAsFloat( float value, params long[] indices )
        {
            if ( indices.Length != DimensionCount ) throw (new ArgumentException( "Number of indices must equal number of tensor dimensions" ));

            for ( int i = 0; i < indices.Length; ++i )
            {
                if ( indices[ i ] < 0 || indices[ i ] >= Sizes[ i ] )
                {
                    throw (new ArgumentException( $"Index {i} with value {indices[ i ]} is out of range" ));
                }
            }

            long offset = 0;
            for ( int i = 0; i < indices.Length; ++i )
            {
                offset += indices[ i ] * _Strides[ i ];
            }
            _Storage.SetElementAsFloat( _StorageOffset + offset, value );
        }
        public void SetElementsAsFloat( float[] value ) => _Storage.SetElementsAsFloat( _StorageOffset, value );

        public void SetElementsAsFloat( float[] value, params long[] indices )
        {
            if ( indices.Length != DimensionCount ) throw (new ArgumentException( "Number of indices must equal number of tensor dimensions" ));

            for ( int i = 0; i < indices.Length; ++i )
            {
                if ( indices[ i ] < 0 || indices[ i ] >= Sizes[ i ] )
                {
                    throw (new ArgumentException( $"Index {i} with value {indices[ i ]} is out of range" ));
                }
            }

            long offset = 0;
            for ( int i = 0; i < indices.Length; ++i )
            {
                offset += indices[ i ] * _Strides[ i ];
            }
            _Storage.SetElementsAsFloat( _StorageOffset + offset, value );
        }

        private string AsText( long[] sizes )
        {
            var sb = new StringBuilder( sizes.Length * (22 + 1) );
            foreach ( long size in sizes )
            {
                sb.Append( size ).Append( ' ' );
            }
            return (sb.ToString());
        }

        public int[] GetElementsAsInt( int length ) => _Storage.GetElementsAsInt( _StorageOffset, length );
        public void SetElementsAsInt( int[] value ) => _Storage.SetElementsAsInt( _StorageOffset, value );

        public Tensor View( params long[] sizes )
        {
            if ( !IsContiguous() ) throw (new InvalidOperationException( "Cannot use View on a non-contiguous tensor" ));
            if ( ElementCount() != TensorDimensionHelpers.ElementCount( sizes ) ) throw (new InvalidOperationException( $"Output tensor must have the same number of elements as the input. Size = {AsText( this._Sizes )}, New Size = {AsText( sizes )}" ));

            return (new Tensor( sizes, TensorDimensionHelpers.GetContiguousStride( sizes ), _Storage, _StorageOffset ));
        }

        public Tensor Narrow( int dimension, long startIndex, long size )
        {
            if ( dimension < 0 || dimension >= DimensionCount ) throw (new ArgumentOutOfRangeException( "dimension" ));
            if ( startIndex < 0 || startIndex >= _Sizes[ dimension ] ) throw (new ArgumentOutOfRangeException( "startIndex", $"startIndex = '{startIndex}', sizes[dimension] = '{_Sizes[ dimension ]}', dimension = '{dimension}', size = '{size}'" ));
            if ( size <= 0 || startIndex + size > _Sizes[ dimension ] ) throw (new ArgumentOutOfRangeException( "size", $"startIndex = '{startIndex}', sizes[dimension] = '{_Sizes[ dimension ]}', dimension = '{dimension}', size = '{size}'" ));

            long newOffset = _StorageOffset + startIndex * _Strides[ dimension ];
            long[] newSizes = _Sizes.ToArray();
            newSizes[ dimension ] = size;

            return (new Tensor( newSizes, _Strides, _Storage, newOffset ));
        }

        public Tensor Select( int dimension, long index )
        {
            if ( DimensionCount == 1 ) throw (new InvalidOperationException( "Select requires 2 or more dimensions" ));
            if ( dimension < 0 || dimension >= DimensionCount ) throw (new ArgumentOutOfRangeException( "dimension" ));
            if ( index < 0 || index >= _Sizes[ dimension ] ) throw (new ArgumentOutOfRangeException( "index" ));

            Tensor result = Narrow( dimension, index, 1 );
            result._Sizes = ArrayRemove( _Sizes, dimension );
            result._Strides = ArrayRemove( _Strides, dimension );

            return (result);
        }

        public Tensor Transpose()
        {
            if ( DimensionCount != 2 ) throw (new InvalidOperationException( "Parameterless Transpose is only valid on 2d tensors" ));
            return (Transpose( 0, 1 ));
        }

        public Tensor Transpose( int dimension1, int dimension2 )
        {
            if ( dimension1 < 0 || dimension1 >= DimensionCount ) throw (new ArgumentOutOfRangeException( "dimension1" ));
            if ( dimension2 < 0 || dimension2 >= DimensionCount ) throw (new ArgumentOutOfRangeException( "dimension2" ));

            long[] newSizes   = _Sizes.ToArray();
            long[] newStrides = _Strides.ToArray();
            ArraySwap( newSizes, dimension1, dimension2 );
            ArraySwap( newStrides, dimension1, dimension2 );
            return (new Tensor( newSizes, newStrides, _Storage, _StorageOffset ));
        }

        public Tensor Permute( params int[] dims )
        {
            if ( dims.Length != DimensionCount ) throw (new InvalidOperationException( "The number of permutation indices must equal the number of tensor dimensions" ));

            Tensor result = CopyRef();
            foreach ( var swap in SwapsForPermutation( dims ) )
            {
                Tensor resultOld = result;
                result = result.Transpose( swap.Item1, swap.Item2 );
                resultOld.Dispose();
            }
            return (result);
        }

        /// <summary>
        /// Expand one or more singleton dimensions (dimensions with size 1) by using a stride of 0
        /// </summary>
        public Tensor Expand( params long[] newSizes )
        {
            if ( newSizes.Length != DimensionCount ) throw (new InvalidOperationException( $"number of elements of newSizes must match the dimension count of tensor. New dimension = '{newSizes.Length}', Current dimension = '{DimensionCount}', New tensor shape = '{string.Join( " ", newSizes )}', current tensor shape = '{string.Join( " ", Sizes )}'" ));

            long[] newStrides = _Strides.ToArray();
            for ( int i = 0; i < newSizes.Length; ++i )
            {
                if ( newSizes[ i ] != Sizes[ i ] )
                {
                    if ( Sizes[ i ] != 1 )
                    {
                        throw (new InvalidOperationException( "Can only expand singleton dimensions (dimensions of size 1)" ));
                    }
                    newStrides[ i ] = 0;
                }
            }
            return (new Tensor( newSizes, newStrides, _Storage, _StorageOffset ));
        }

        /// <summary>
        /// Return a new tensor where **all** singleton dimensions have been removed
        /// </summary>
        public Tensor Squeeze()
        {
            Tuple<long, long>[] newSizeStrides = _Sizes.Zip( _Strides, Tuple.Create )
                .Where( x => x.Item1 != 1 )
                .ToArray();

            long[] newSizes   = newSizeStrides.Select( x => x.Item1 ).ToArray();
            long[] newStrides = newSizeStrides.Select( x => x.Item2 ).ToArray();

            return (new Tensor( newSizes, newStrides, _Storage, _StorageOffset ));
        }

        /// <summary>
        /// Return a new tensor where the given singleton dimension has been removed
        /// </summary>
        public Tensor Squeeze( int dimension )
        {
            if ( DimensionCount == 1 ) throw (new InvalidOperationException( "Squeeze requires 2 or more dimensions" ));
            if ( dimension < 0 || dimension >= DimensionCount ) throw (new ArgumentOutOfRangeException( "dimension" ));

            long[] newSizes   = ArrayRemove( _Sizes, dimension );
            long[] newStrides = ArrayRemove( _Strides, dimension );

            return (new Tensor( newSizes, newStrides, _Storage, _StorageOffset ));
        }

        /// <summary>
        /// Returns a tensor which contains all slices of size size in the given dimension. The step between two slices is given by step.
        /// The result tensor has an additional dimension of size size.
        /// </summary>
        public Tensor Unfold( int dimension, long size, long step )
        {
            if ( DimensionCount == 0 ) throw (new InvalidOperationException( "Cannot unfold an empty tensor" ));
            if ( dimension < 0 || dimension >= DimensionCount ) throw (new ArgumentOutOfRangeException( "dimension is out of range", "dimension" ));
            if ( size > _Sizes[ dimension ] ) throw (new ArgumentOutOfRangeException( "size cannot be larger than the size of dimension", "size" ));
            if ( step <= 0 ) throw (new ArgumentOutOfRangeException( "step must be at least 1", "step" ));

            var newSize    = new long[ DimensionCount + 1 ];
            var newStrides = new long[ DimensionCount + 1 ];
            Array.Copy( _Sizes, newSize, DimensionCount );
            Array.Copy( _Strides, newStrides, DimensionCount );

            newSize   [ DimensionCount ] = size;
            newStrides[ DimensionCount ] = _Strides[ dimension ];

            newSize   [ dimension ] = (_Sizes[ dimension ] - size) / step + 1;
            newStrides[ dimension ] = step * _Strides[ dimension ];

            return (new Tensor( newSize, newStrides, Storage, StorageOffset ));
        }

        // Pad array by prepending with 1 until its length equals newSize
        private static long[] Pad1Prepend( long[] array, int newSize )
        {
            var result = new long[ newSize ];
            // Fill new extra elements with 1
            for ( int i = 0; i < newSize - array.Length; ++i )
            {
                result[ i ] = 1;
            }
            // Copy array to the last array.Length elements of result
            Array.Copy( array, 0, result, newSize - array.Length, array.Length );
            return (result);
        }

        // Prepend singleton dimensions until DimensionCount equals newDimCount
        private Tensor PadToDimCount( int newDimCount )
        {
            long[] newSizes   = Pad1Prepend( _Sizes, newDimCount );
            long[] newStrides = TensorDimensionHelpers.GetContiguousStride( newSizes );
            Array.Copy( _Strides, 0, newStrides, newStrides.Length - _Strides.Length, _Strides.Length );

            return (new Tensor( newSizes, newStrides, _Storage, _StorageOffset ));
        }

        public Tensor RepeatTensor( params long[] repetitions )
        {
            if ( repetitions.Length < DimensionCount ) throw (new InvalidOperationException( "repetitions must be at least the same length as the number of tensor dimensions" ));
            if ( repetitions.Any( x => x < 1 ) ) throw (new InvalidOperationException( "All dimensions must be repeated at least once" ));

            Tensor paddedSrc = PadToDimCount( repetitions.Length );
            long[] resultSize = paddedSrc.Sizes.Zip( repetitions, ( s, r ) => s * r ).ToArray();

            var result = new Tensor( Allocator, ElementType, resultSize );

            Tensor urTensor = result.CopyRef();
            for ( int i = 0; i < paddedSrc.DimensionCount; ++i )
            {
                Tensor oldUrTensor = urTensor;
                urTensor = urTensor.Unfold( i, paddedSrc.Sizes[ i ], paddedSrc.Sizes[ i ] );
                oldUrTensor.Dispose();
            }

            Tensor paddedSrc2 = paddedSrc.PadToDimCount( urTensor.DimensionCount );
            Tensor expandedSrc = paddedSrc2.Expand( urTensor.Sizes );
            Ops.Copy( urTensor, expandedSrc );

            paddedSrc.Dispose();
            paddedSrc2.Dispose();
            urTensor.Dispose();
            expandedSrc.Dispose();

            return (result);
        }

        public void CopyFrom( Array array )
        {
            DType elementType = DTypeBuilder.FromCLRType( array.GetType().GetElementType() );

            if ( !IsContiguous() ) throw (new InvalidOperationException( "Tensor must be contiguous to copy from CLR array" ));
            if ( ElementCount() != array.LongLength ) throw (new InvalidOperationException( "Tensor and array must have the same number of elements" ));
            if ( ElementType != elementType ) throw (new InvalidOperationException( "Tensor and array must have the same element types" ));

            var handle = GCHandle.Alloc( array, GCHandleType.Pinned );
            try
            {
                int length = Buffer.ByteLength( array );
                Storage.CopyToStorage( StorageOffset, handle.AddrOfPinnedObject(), length );
            }
            finally
            {
                handle.Free();
            }
        }

        public void CopyToArray( Array array )
        {
            DType elementType = DTypeBuilder.FromCLRType( array.GetType().GetElementType() );

            if ( !IsContiguous() ) throw (new InvalidOperationException( "Tensor must be contiguous to copy from CLR array" ));
            if ( ElementCount() != array.LongLength ) throw (new InvalidOperationException( "Tensor and array must have the same number of elements" ));
            if ( ElementType != elementType ) throw (new InvalidOperationException( "Tensor and array must have the same element types" ));

            var handle = GCHandle.Alloc( array, GCHandleType.Pinned );
            try
            {
                int length = Buffer.ByteLength( array );
                Storage.CopyFromStorage( handle.AddrOfPinnedObject(), StorageOffset, length );
            }
            finally
            {
                handle.Free();
            }
        }

        public static Tensor FromArray( IAllocator allocator, Array array )
        {
            // From the CLI spec(section 8.9.1):
            // Array elements shall be laid out within the array object in row - major order
            // (i.e., the elements associated with the rightmost array dimension shall be laid out contiguously from lowest to highest index).
            // The actual storage allocated for each array element can include platform - specific padding.

            // This is already in the order we want - and here we will (potentially incorrectly) assume that there is no
            // 'platform-specific padding'. This appears to be a reasonable assumption on both CLR and Mono.
            // Assuming no platform-specific padding allows us to use memcpy instead of iterating and copying each element

            DType elementType = DTypeBuilder.FromCLRType( array.GetType().GetElementType() );

            long[] dimSizes = Enumerable.Range( 0, array.Rank ).Select( x => (long) array.GetLength( x ) ).ToArray();

            Tensor result = new Tensor( allocator, elementType, dimSizes );
            result.CopyFrom( array );
            return (result);
        }

        [M(O.AggressiveInlining)] private static void ArraySwap< T >( T[] array, int index1, int index2 )
        {
            T t = array[ index1 ];
            array[ index1 ] = array[ index2 ];
            array[ index2 ] = t;
        }

        // Return a copy of an array, but with the item at index removed
        [M(O.AggressiveInlining)] private static T[] ArrayRemove< T >( T[] source, long index )
        {
            var result = new T[ source.Length - 1 ];
            for ( int i = 0; i < result.Length; ++i )
            {
                if ( i < index )
                {
                    result[ i ] = source[ i ];
                }
                else
                {
                    result[ i ] = source[ i + 1 ];
                }
            }
            return (result);
        }

        // Convert a permutation into a sequence of swap operations.
        // perm must contain a permuation of the indices [0, perm.Length)
        // The returned tuples indicate pairs of indices that should be swapped. The swaps
        // must be performed in the given order.
        [M(O.AggressiveInlining)] private static IEnumerable< (int, int) > SwapsForPermutation( int[] perm )
        {
            int j;
            for ( int i = 0; i < perm.Length; ++i )
            {
                int p = perm[ i ];
                if ( p != i && p != -1 )
                {
                    j = i;
                    do
                    {
                        if ( perm[ j ] < 0 || perm[ j ] >= perm.Length )
                        {
                            throw (new InvalidOperationException( "Invalid permutation" ));
                        }

                        yield return (j, perm[ j ]);

                        int jOld = j;
                        j = perm[ j ];
                        perm[ jOld ] = -1;
                    }
                    while ( perm[ j ] != i );
                    perm[ j ] = j;
                }
            }
        }

        public float[] ToWeightArray() => this.GetElementsAsFloat( (int) this.GetStorageSize() );

        public override string ToString() => TensorFormatting.FormatTensorTypeAndSize( this );
    }
}
