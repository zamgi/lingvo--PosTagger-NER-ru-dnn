using System;
using System.Linq;
using System.Text;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    internal static class TensorFormatting
    {
        private static string RepeatChar( char c, int count )
        {
            var sb = new StringBuilder( count );
            for ( int i = 0; i < count; ++i )
            {
                sb.Append( c );
            }
            return (sb.ToString());
        }

        private static string GetIntFormat( int length )
        {
            var padding = RepeatChar( '#', length - 1 );
            return string.Format( " {0}0;-{0}0", padding );
        }
        private static string GetFloatFormat( int length )
        {
            var padding = RepeatChar( '#', length - 1 );
            return string.Format( " {0}0.0000;-{0}0.0000", padding );
        }
        private static string GetScientificFormat( int length )
        {
            var padCount = length - 6;
            var padding = RepeatChar( '0', padCount );
            return string.Format( " {0}.0000e+00;-0.{0}e+00", padding );
        }

        private static bool IsIntOnly( Storage storage, Tensor tensor )
        {
            // HACK this is a hacky way of iterating over the elements of the tensor.
            // if the tensor has holes, this will incorrectly include those elements
            // in the iteration.
            long minOffset = tensor.StorageOffset;
            long maxOffset = minOffset + TensorDimensionHelpers.GetStorageSize( tensor.Sizes, tensor.Strides ) - 1;
            for ( long i = minOffset; i <= maxOffset; ++i )
            {
                double value = Convert.ToDouble( (object) storage.GetElementAsFloat( i ) );
                if ( value != Math.Ceiling( value ) )
                {
                    return (false);
                }
            }
            return (true);
        }

        private static Tuple<double, double> AbsMinMax( Storage storage, Tensor tensor )
        {
            if ( storage.ElementCount == 0 )
            {
                return Tuple.Create( 0.0, 0.0 );
            }

            double min = storage.GetElementAsFloat( 0 );
            double max = storage.GetElementAsFloat( 0 );

            // HACK this is a hacky way of iterating over the elements of the tensor.
            // if the tensor has holes, this will incorrectly include those elements
            // in the iteration.
            long minOffset = tensor.StorageOffset;
            long maxOffset = minOffset + TensorDimensionHelpers.GetStorageSize( tensor.Sizes, tensor.Strides ) - 1;

            for ( long i = minOffset; i <= maxOffset; ++i )
            {
                float item = storage.GetElementAsFloat( i );
                if ( item < min )
                {
                    min = item;
                }

                if ( item > max )
                {
                    max = item;
                }
            }

            return Tuple.Create( Math.Abs( min ), Math.Abs( max ) );
        }

        /// <summary>
        /// 
        /// </summary>
        private enum FormatType
        {
            Int,
            Scientific,
            Float,
        }
        private static Tuple<FormatType, double, int> GetFormatSize( Tuple<double, double> minMax, bool intMode )
        {
            int expMin = (minMax.Item1 != 0) ? (int) Math.Floor( Math.Log10( minMax.Item1 ) ) + 1 : 1;
            int expMax = (minMax.Item2 != 0) ? (int) Math.Floor( Math.Log10( minMax.Item2 ) ) + 1 : 1;

            if ( intMode )
            {
                if ( expMax > 9 )
                {
                    return Tuple.Create( FormatType.Scientific, 1.0, 11 );
                }
                else
                {
                    return Tuple.Create( FormatType.Int, 1.0, expMax + 1 );
                }
            }
            else
            {
                if ( expMax - expMin > 4 )
                {
                    int sz = (Math.Abs( expMax ) > 99 || Math.Abs( expMin ) > 99) ? 12 : 11;
                    return Tuple.Create( FormatType.Scientific, 1.0, sz );
                }
                else
                {
                    if ( expMax > 5 || expMax < 0 )
                    {
                        return Tuple.Create( FormatType.Float, Math.Pow( 10, expMax - 1 ), 7 );
                    }
                    else
                    {
                        return Tuple.Create( FormatType.Float, 1.0, expMax == 0 ? 7 : expMax + 6 );
                    }
                }
            }
        }

        private static string BuildFormatString( FormatType type, int size )
        {
            switch ( type )
            {
                case FormatType.Int:        return GetIntFormat( size );
                case FormatType.Float:      return GetFloatFormat( size );
                case FormatType.Scientific: return GetScientificFormat( size );
                default: throw (new InvalidOperationException( $"Invalid format type '{type}'." ));
            }
        }

        private static Tuple<string, double, int> GetStorageFormat( Storage storage, Tensor tensor )
        {
            if ( storage.ElementCount == 0 )
            {
                return Tuple.Create( "", 1.0, 0 );
            }

            bool intMode = IsIntOnly( storage, tensor );
            Tuple<double, double> minMax = AbsMinMax( storage, tensor );

            Tuple<FormatType, double, int> formatSize = GetFormatSize( minMax, intMode );
            string formatString = BuildFormatString( formatSize.Item1, formatSize.Item3 );

            return Tuple.Create( "{0:" + formatString + "}", formatSize.Item2, formatSize.Item3 );
        }

        public static string FormatTensorTypeAndSize( Tensor tensor )
        {
            var sb = new StringBuilder().Append( '[' ).Append( tensor.ElementType ).Append( " tensor" );
            if ( tensor.DimensionCount == 0 )
            {
                sb.Append( " with no dimension" );
            }
            else
            {
                sb.Append( " of size " ).Append( tensor.Sizes[ 0 ] );
                for ( int i = 1; i < tensor.DimensionCount; ++i )
                {
                    sb.Append( "x" ).Append( tensor.Sizes[ i ] );
                }
            }
            sb.Append( " on " ).Append( tensor.Storage.LocationDescription() );
            sb.Append( "]" );
            return (sb.ToString());
        }

        private static void FormatVector( StringBuilder sb, Tensor tensor )
        {
            Tuple<string, double, int> storageFormat = GetStorageFormat( tensor.Storage, tensor );
            string format = storageFormat.Item1;
            double scale = storageFormat.Item2;

            if ( scale != 1 )
            {
                sb.AppendLine( scale + " *" );
                for ( int i = 0; i < tensor.Sizes[ 0 ]; ++i )
                {
                    double value = Convert.ToDouble( (object) tensor.GetElementAsFloat( i ) ) / scale;
                    sb.AppendLine( string.Format( format, value ) );
                }
            }
            else
            {
                for ( int i = 0; i < tensor.Sizes[ 0 ]; ++i )
                {
                    double value = Convert.ToDouble( (object) tensor.GetElementAsFloat( i ) );
                    sb.AppendLine( string.Format( format, value ) );
                }
            }
        }
        private static void FormatMatrix( StringBuilder sb, Tensor tensor, string indent )
        {
            Tuple<string, double, int> storageFormat = GetStorageFormat( tensor.Storage, tensor );
            string format = storageFormat.Item1;
            double scale = storageFormat.Item2;
            int sz = storageFormat.Item3;

            sb.Append( indent );

            int nColumnPerLine = (int) Math.Floor( (80 - indent.Length) / (double) (sz + 1) );
            long firstColumn = 0;
            while ( firstColumn < tensor.Sizes[ 1 ] )
            {
                long lastColumn;
                if ( firstColumn + nColumnPerLine - 2 < tensor.Sizes[ 1 ] )
                {
                    lastColumn = firstColumn + nColumnPerLine - 2;
                }
                else
                {
                    lastColumn = tensor.Sizes[ 1 ] - 1;
                }

                if ( nColumnPerLine < tensor.Sizes[ 1 ] )
                {
                    if ( firstColumn != 1 )
                    {
                        sb.AppendLine();
                    }
                    sb.Append( "Columns " ).Append( firstColumn ).Append( " to " ).Append( lastColumn ).AppendLine();
                }

                if ( scale != 1 )
                {
                    sb.Append( scale ).AppendLine( " *" );
                }

                for ( long l = 0; l < tensor.Sizes[ 0 ]; ++l )
                {
                    using ( Tensor row = tensor.Select( 0, l ) )
                    {
                        for ( long c = firstColumn; c <= lastColumn; ++c )
                        {
                            double value = Convert.ToDouble( (object) row.GetElementAsFloat( c ) ) / scale;
                            sb.Append( string.Format( format, value ) );
                            if ( c == lastColumn )
                            {
                                sb.AppendLine();
                                if ( l != tensor.Sizes[ 0 ] )
                                {
                                    sb.Append( scale != 1 ? indent + " " : indent );
                                }
                            }
                            else
                            {
                                sb.Append( ' ' );
                            }
                        }
                    }
                }
                firstColumn = lastColumn + 1;
            }
        }

        private static void FormatTensor( StringBuilder builder, Tensor tensor )
        {
            var startingLength = builder.Length;
            var counter = Enumerable.Repeat( (long) 0, tensor.DimensionCount - 2 ).ToArray();
            var finished = false;
            counter[ 0 ] = -1;
            while ( true )
            {
                for ( int i = 0; i < tensor.DimensionCount - 2; ++i )
                {
                    counter[ i ]++;
                    if ( counter[ i ] >= tensor.Sizes[ i ] )
                    {
                        if ( i == tensor.DimensionCount - 3 )
                        {
                            finished = true;
                            break;
                        }
                        counter[ i ] = 1;
                    }
                    else
                    {
                        break;
                    }
                }

                if ( finished )
                {
                    break;
                }

                if ( builder.Length - startingLength > 1 )
                {
                    builder.AppendLine();
                }

                builder.Append( '(' );
                Tensor tensorCopy = tensor.CopyRef();
                for ( int i = 0; i < tensor.DimensionCount - 2; ++i )
                {
                    Tensor newCopy = tensorCopy.Select( 0, counter[ i ] );
                    tensorCopy.Dispose();
                    tensorCopy = newCopy;
                    builder.Append( counter[ i ] ).Append( ',' );
                }

                builder.AppendLine( ".,.) = " );
                FormatMatrix( builder, tensorCopy, " " );

                tensorCopy.Dispose();
            }
        }

        public static string Format( Tensor tensor )
        {
            var sb = new StringBuilder();
            if ( tensor.DimensionCount == 0 )
            {
                ;
            }
            else if ( tensor.DimensionCount == 1 )
            {
                FormatVector( sb, tensor );
            }
            else if ( tensor.DimensionCount == 2 )
            {
                FormatMatrix( sb, tensor, "" );
            }
            else
            {
                FormatTensor( sb, tensor );
            }
            sb.AppendLine( FormatTensorTypeAndSize( tensor ) );
            return (sb.ToString());
        }
    }
}
