using System;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal static class PositionEmbedding
    {
        public static WeightTensor AddPositionEmbedding( ComputeGraphTensor g, WeightTensor posEmbedding, int batchSize, WeightTensor inputEmbs, float dropoutRatio )
        {
            var column = posEmbedding.Columns;
            int seqLen = inputEmbs.Rows / batchSize;

            using ( var posEmbeddingPeek        = g.Peek( posEmbedding, 0, 0, seqLen ) )
            using ( var posEmbeddingPeekView    = g.View( posEmbeddingPeek, dims: new long[] { 1, seqLen, column } ) )
            using ( var posEmbeddingPeekViewExp = g.Expand( posEmbeddingPeekView, dims: new long[] { batchSize, seqLen, column } ) )
            {
                inputEmbs = g.View( inputEmbs, dims: new long[] { batchSize, seqLen, column } );
                inputEmbs = g.Add ( inputEmbs, posEmbeddingPeekViewExp, inPlace: true );
                inputEmbs = g.View( inputEmbs, dims: new long[] { batchSize * seqLen, column } );
            }

            inputEmbs = g.Dropout( inputEmbs, batchSize, dropoutRatio, inPlace: true );
            return (inputEmbs);
        }

        public static WeightTensor BuildPositionWeightTensor( int row, int column, int deviceId, string name, bool isTrainable = false )
        {
            Logger.WriteLine( $"Building position weights tensor. Row = '{row}', Column = '{column}', DeviceId = '{deviceId}', Name = '{name}', Trainable = '{isTrainable}'" );

            var wt = new WeightTensor( new long[ 2 ] { row, column }, deviceId, name: name, isTrainable: isTrainable, needGradient: isTrainable );
            var posWeights = new float[ row * column ];

            float numTimescales         = ((float) column) / 2;
            float logTimescaleIncrement = MathF.Log( 10000.0f ) / (numTimescales - 1.0f);

            for ( var r = 0; r < row; ++r )
            {
                for ( var i = 0; i < numTimescales; i++ )
                {
                    float v = (float) (r * Math.Exp( i * -logTimescaleIncrement ));

                    posWeights[ r * column + i ] = MathF.Sin( v );
                    posWeights[ r * column + (int) numTimescales + i ] = MathF.Cos( v );
                }
            }

            wt.TWeight.CopyFrom( posWeights );
            return (wt);
        }
    }
}
