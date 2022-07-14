using System;

namespace Lingvo.PosTagger.Tensors
{
    public static class RandomGenerator
    {
        private static Random _Rnd;
        static RandomGenerator() => _Rnd = new Random( DateTime.Now.Millisecond );

        public static float[] BuildRandomUniformWeight( long[] sizes, float min, float max )
        {
            long size = 1;
            foreach ( var s in sizes )
            {
                size *= s;
            }

            var w = new float[ size ];
            for ( int i = 0; i < size; i++ )
            {
                w[ i ] = (float) _Rnd.NextDouble() * (max - min) + min;
            }
            return w;
        }
        public static float[] BuildRandomBernoulliWeight( long[] sizes, float p )
        {
            long size = 1;
            foreach ( var s in sizes )
            {
                size *= s;
            }

            var w = new float[ size ];
            for ( int i = 0; i < size; i++ )
            {
                w[ i ] = _Rnd.NextDouble() <= p ? 1.0f : 0.0f;
            }
            return w;
        }
    }
}
