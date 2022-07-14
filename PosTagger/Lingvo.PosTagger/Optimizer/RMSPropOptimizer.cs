using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using Lingvo.PosTagger.Network;
using Lingvo.PosTagger.Tensors;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Optimizer
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class RMSPropOptimizer : IOptimizer
    {
        private const float _SmoothEps = 1e-9f;

        private ConcurrentDictionary< string, Tensor > _CacheName2V;
        private float _GradClip;
        private float _DecayRate;

        public RMSPropOptimizer( float gradClip, float decayRate = 0.999f )
        {
            Logger.WriteLine( $"Creating RMSProp optimizer. GradClip = '{gradClip}', LR decay rate = '{decayRate}'" );

            _CacheName2V = new ConcurrentDictionary< string, Tensor >();
            _GradClip    = gradClip;
            _DecayRate   = decayRate;
        }

        public void UpdateWeights( List< WeightTensor > model, int batchSize, float step_size, float regc, int iter )
        {
            var id2Models   = new Dictionary< int, List< WeightTensor > >();
            var name2tensor = new Dictionary< string, WeightTensor >();

            foreach ( WeightTensor wt in model )
            {
                if ( !wt.IsTrainable )
                {
                    continue;
                }

                if ( name2tensor.TryGetValue( wt.Name, out var v ) )
                {
                    if ( wt != v )
                    {
                        throw (new ArgumentException( $"Found duplicated weights '{wt.Name}'." ));
                    }
                    continue;
                }
                name2tensor.Add( wt.Name, wt );

                if ( !id2Models.TryGetValue( wt.DeviceId, out var lst ) )
                {
                    lst = new List< WeightTensor >();
                    id2Models.Add( wt.DeviceId, lst );
                }
                lst.Add( wt );

                if ( !_CacheName2V.ContainsKey( wt.Name ) )
                {
                    var t = new Tensor( wt.Allocator, DType.Float32, wt.Sizes );
                    _CacheName2V[ wt.Name ] = t;
                    Ops.Fill( t, 0.0f );

                    //---Logger.WriteLine( $"Added weight '{wt.Name}' to optimizer." );
                }
            }

            Parallel.ForEach( id2Models.Values, lst =>
            {
                 foreach ( WeightTensor wt in lst )
                 {
                     UpdateWeightsTensor( wt, batchSize, step_size, regc );
                 }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] private void UpdateWeightsTensor( WeightTensor wt, int batchSize, float step_size, float regc )
        {
            try
            {
                if ( !_CacheName2V.TryGetValue( wt.Name, out var v ) )
                {
                    throw (new KeyNotFoundException( $"!_CacheName2V[ wt.Name ] => wt.Name='{wt.Name}'" ));
                }

                Ops.RMSProp( wt.TWeight, wt.TGradient, v, batchSize, step_size, _GradClip, regc, _DecayRate, _SmoothEps );
            }
            catch ( Exception ex )
            {
                Logger.WriteErrorLine( $"Exception: '{ex.Message}', Call stack: '{ex.StackTrace}'" );
                throw;
            }
        }
    }
}
