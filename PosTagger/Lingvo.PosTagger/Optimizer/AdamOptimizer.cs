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
    internal sealed class AdamOptimizer : IOptimizer
    {
        private const float _SmoothEps = 1e-9f;

        private ConcurrentDictionary< string, Tensor > _CacheName2V;
        private ConcurrentDictionary< string, Tensor > _CacheName2M;
        private float _Beta1;
        private float _Beta2;
        private float _GradClip;

        public AdamOptimizer( float gradClip, float beta1 = 0.9f, float beta2 = 0.98f )
        {
            Logger.WriteLine( $"Creating Adam optimizer. GradClip = '{gradClip}', Beta1 = '{beta1}', Beta2 = '{beta2}'" );

            _CacheName2V = new ConcurrentDictionary< string, Tensor >();
            _CacheName2M = new ConcurrentDictionary< string, Tensor >();

            _GradClip = gradClip;
            _Beta1    = beta1;
            _Beta2    = beta2;
        }

        public void UpdateWeights( List< WeightTensor > model, int batchSize, float step_size, float regc, int iter )
        {
            var id2Models   = new Dictionary< int, List< WeightTensor > >( model.Count );
            var name2tensor = new Dictionary< string, WeightTensor >( model.Count );

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
                    lst = new List<WeightTensor>( model.Count );
                    id2Models.Add( wt.DeviceId, lst );
                }
                lst.Add( wt );

                if ( !_CacheName2V.ContainsKey( wt.Name ) )
                {
                    //---Logger.WriteLine( $"Begin adding weight '{wt.Name}' to optimizer (_CacheName2V)..." );

                    var t = new Tensor( wt.Allocator, DType.Float32, wt.Sizes );
                    _CacheName2V[ wt.Name ] = t;
                    Ops.Fill( t, 0.0f );

                    //---Logger.WriteLine( $"Added weight '{wt.Name}' to optimizer (_CacheName2V). Learning rate factor = '{wt.LearningRateFactor}'" );

                    //---Logger.WriteLine( $"Begin adding weight '{wt.Name}' to optimizer (_CacheName2M)..." );

                    t = new Tensor( wt.Allocator, DType.Float32, wt.Sizes );
                    _CacheName2M[ wt.Name ] = t;
                    Ops.Fill( t, 0.0f );

                    //---Logger.WriteLine( $"Added weight '{wt.Name}' to optimizer (_CacheName2M). Learning rate factor = '{wt.LearningRateFactor}'" );

                    //---//---Logger.WriteLine( $"Added weight '{wt.Name}' to optimizer. Learning rate factor = '{wt.LearningRateFactor}'" );
                }
            }

            Parallel.ForEach( id2Models.Values, lst =>
            {
                 foreach ( WeightTensor wt in lst )
                 {
                     UpdateWeightsTensor( wt, batchSize, step_size * wt.LearningRateFactor, regc, iter );
                 }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] private void UpdateWeightsTensor( WeightTensor wt, int batchSize, float step_size, float regc, int iter )
        {
            try
            {
                if ( !_CacheName2V.TryGetValue( wt.Name, out var v ) )
                {
                    throw (new KeyNotFoundException( $"!_CacheName2V[ wt.Name ] => wt.Name='{wt.Name}'" ));
                }
                if ( !_CacheName2M.TryGetValue( wt.Name, out var m ) )
                {
                    throw (new KeyNotFoundException( $"!_CacheName2M[ wt.Name ] => wt.Name='{wt.Name}'" ));
                }

                Ops.Adam( wt.TWeight, wt.TGradient, v, m/*, _CacheName2V[ wt.Name ], _CacheName2M[ wt.Name ]*/, batchSize, step_size, _GradClip, regc, _Beta2, _Beta1, iter, _SmoothEps );
            }
            catch ( Exception ex )
            {
                Logger.WriteErrorLine( $"Exception: '{ex.Message}', Call stack: '{ex.StackTrace}'" );
                throw;
            }
        }
    }
}
