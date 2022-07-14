//#define USE_VISUALIZE_NETWORK

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

using Lingvo.PosTagger.Tensors;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class ConcurrentList< T >
    {
        private const int MAX_SIZE = 1_024_000;
        private T[] _Array;
        private int _Count;

        public ConcurrentList() => _Array = new T[ MAX_SIZE ];

        public int Count => _Count;
        public T this[ int key ] => _Array[ key ];

        public void Add( T t )
        {
            int n = Interlocked.Increment( ref _Count );
            _Array[ n - 1 ] = t;
        }
        public void Clear()
        {
            Interlocked.Exchange( ref _Count, 0 );
            Interlocked.Exchange( ref _Array, null );
        }
    }
    
    /// <summary>
    /// 
    /// </summary>
    public sealed class ComputeGraphTensor : IDisposable
    {
        private readonly WeightTensorFactory _WeightTensorFactory;
        private readonly ConcurrentList<Action> _BackProp;
        private readonly bool _NeedsBackProp;
        private readonly int _DeviceId;
        private readonly bool _IsSubGraph;
        private readonly List< WeightTensor > _TensorsBindToCurrentGraph;
#if USE_VISUALIZE_NETWORK
        // Visualization for neural network
        private bool _VisualizeNetwork;
        private Microsoft.Msagl.Drawing.Graph _MsGraph;
        private HashSet<string> _MsGraph_Edges;
        private Microsoft.Msagl.Drawing.Subgraph _MsSubGraph;
        private Dictionary<string, Microsoft.Msagl.Drawing.Subgraph> _Name2SubGraph;
#endif
        internal ComputeGraphTensor( int deviceId, bool needBack = true, ConcurrentList< Action > backprop = null, bool isSubGraph = false ) 
            : this( new WeightTensorFactory(), deviceId, needBack, backprop, isSubGraph ) { }
        private ComputeGraphTensor( WeightTensorFactory weightFactory, int deviceId, bool needBack, ConcurrentList< Action > backprop, bool isSubGraph )
        {
            _BackProp                  = backprop ?? new ConcurrentList< Action >();
            _WeightTensorFactory       = weightFactory;
            _NeedsBackProp             = needBack;
            _DeviceId                  = deviceId;
            _IsSubGraph                = isSubGraph;
            _TensorsBindToCurrentGraph = new List< WeightTensor >();
#if USE_VISUALIZE_NETWORK
            _VisualizeNetwork = visualizeNetwork;            
            if ( _VisualizeNetwork )
            {
                // Initialize parameters for neural network visualization
                _MsGraph = new Microsoft.Msagl.Drawing.Graph();
                _MsGraph_Edges = new HashSet<string>();
                _Name2SubGraph = new Dictionary<string, Subgraph>();
            }
#endif
        }

        public void Dispose()
        {
            // We only dispose root computing graph, For sub graph, we don't do it.
            if ( !_IsSubGraph )
            {
                if ( _BackProp != null )
                {
                    _BackProp.Clear();
                }

                if ( _WeightTensorFactory != null )
                {
                    _WeightTensorFactory.Dispose();
                }
            }
            else
            {
                foreach ( WeightTensor wt in _TensorsBindToCurrentGraph )
                {
                    wt.ReleaseWeight();
                }
            }

            _TensorsBindToCurrentGraph.Clear();
        }

        public WeightTensorFactory GetWeightFactory() => _WeightTensorFactory;
        public ComputeGraphTensor CreateSubGraph( string name )
        {
            var subGraph = new ComputeGraphTensor( _WeightTensorFactory, _DeviceId, _NeedsBackProp, _BackProp, isSubGraph: true );
            //if ( _VisualizeNetwork )
            //{
            //    // Create parameters for neural network visualization
            //    subGraph._MsGraph = _MsGraph;
            //    subGraph._MsGraph_Edges = _MsGraph_Edges;
            //    subGraph._Name2SubGraph = _Name2SubGraph;
            //    if (!_Name2SubGraph.ContainsKey(name))
            //    {
            //        int index = name.LastIndexOf(".");
            //        subGraph._MsSubGraph = new Subgraph(name) { LabelText = name.Substring(index + 1) };

            //        _Name2SubGraph.Add(name, subGraph._MsSubGraph);

            //        if (_MsSubGraph == null)
            //        {
            //            _MsGraph.RootSubgraph.AddSubgraph(subGraph._MsSubGraph);
            //        }
            //        else
            //        {
            //            _MsSubGraph.AddSubgraph(subGraph._MsSubGraph);
            //        }
            //    }
            //    else
            //    {
            //        subGraph._MsSubGraph = _Name2SubGraph[name];
            //    }
            //}

            return (subGraph);
        }

        public void Backward()
        {
            for ( int i = _BackProp.Count - 1; i >= 0; i-- )
            {
                _BackProp[ i ](); // tick!
            }
            _BackProp.Clear();
        }

        public WeightTensor Sigmoid( WeightTensor w )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Sigmoid", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Sigmoid( wt.TWeight, w.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        w.AddSigmoidGradient( wt );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }
        public WeightTensor Rsqrt( WeightTensor w )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Rsqrt", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Rsqrt( wt.TWeight, w.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        using ( var tmp = Ops.Pow( null, wt.TWeight, 3.0f ) )
                        {
                            using var tmp2 = Ops.Mul( null, tmp, wt.TGradient );
                            using var tmp3 = Ops.Mul( null, tmp2, -0.5f );
                            w.CopyOrAddGradient( tmp3 );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }
        public WeightTensor AddTanh( WeightTensor w1, WeightTensor w2 )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w1.Sizes, _DeviceId, name: $"{GetHashString( w1.Name, w2.Name )}.AddTanh", graphToBind: this, needGradient: (w1.NeedGradient || w2.NeedGradient) );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w1, w2 }, wt );
#endif
            Ops.AddTanh( wt.TWeight, w1.TWeight, w2.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w1.NeedGradient )
                    {
                        w1.AddTanhGradient( wt );
                    }

                    if ( w2.NeedGradient )
                    {
                        w2.AddTanhGradient( wt );
                    }

                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor AddTanh( WeightTensor w1, WeightTensor w2, WeightTensor w3 )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w1.Sizes, _DeviceId, name: $"{GetHashString( w1.Name, w2.Name, w3.Name )}.AddTanh", graphToBind: this, needGradient: (w1.NeedGradient || w2.NeedGradient || w3.NeedGradient) );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w1, w2, w3 }, wt );
#endif
            Ops.AddTanh3( wt.TWeight, w1.TWeight, w2.TWeight, w3.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w1.NeedGradient )
                    {
                        w1.AddTanhGradient( wt );
                    }

                    if ( w2.NeedGradient )
                    {
                        w2.AddTanhGradient( wt );
                    }

                    if ( w3.NeedGradient )
                    {
                        w3.AddTanhGradient( wt );
                    }

                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Mul( WeightTensor w, float v, bool inPlace = false )
        {
            WeightTensor wt;
            if ( inPlace )
            {
                wt = w.CopyWeightsRef( $"{GetHashString( w.Name )}.MulV", w.NeedGradient );
            }
            else
            {
                wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.MulV", graphToBind: this, needGradient: w.NeedGradient );
            }
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Mul( wt.TWeight, w.TWeight, v );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();

                        if ( inPlace && wt.TGradient.IsOwnerExclusive() && w.IsGradientNull() )
                        {
                            w.TGradient = wt.TGradient.CopyRef();
                            Ops.Mul( w.TGradient, wt.TGradient, v );
                        }
                        else
                        {
                            Ops.AddMulV( w.TGradient, w.TGradient, wt.TGradient, v );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }


        public WeightTensor Div( WeightTensor w, float v, bool inPlace = false )
        {            
            WeightTensor wt;
            if ( inPlace )
            {
                wt = w.CopyWeightsRef( $"{GetHashString( w.Name )}.DivV", w.NeedGradient );
            }
            else
            {
                wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.DivV", graphToBind: this, needGradient: w.NeedGradient );
            }
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Div( wt.TWeight, w.TWeight, v );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();

                        if ( inPlace && wt.TGradient.IsOwnerExclusive() && w.IsGradientNull() )
                        {
                            w.TGradient = wt.TGradient.CopyRef();
                            Ops.Div( w.TGradient, wt.TGradient, v );
                        }
                        else
                        {
                            Ops.AddMulV( w.TGradient, w.TGradient, wt.TGradient, 1.0f / v );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public void Bind( WeightTensor w ) => _TensorsBindToCurrentGraph.Add( w );
        public void Unbind( WeightTensor w ) => _TensorsBindToCurrentGraph.Remove( w );

        /// <summary>
        /// Result = w1 * w2 + w3 * w4
        /// </summary>
        public WeightTensor EltMulMulAdd( WeightTensor w1, WeightTensor w2, WeightTensor w3, WeightTensor w4 )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w1.Sizes, _DeviceId, name: $"{GetHashString( w1.Name, w2.Name, w3.Name, w4.Name )}.EltMulMulAdd", graphToBind: this, needGradient: (w1.NeedGradient || w2.NeedGradient || w3.NeedGradient || w4.NeedGradient) );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w1, w2, w3, w4 }, wt );
#endif
            Ops.MulMulAdd( wt.TWeight, w1.TWeight, w2.TWeight, w3.TWeight, w4.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    wt.ReleaseWeight();
                    if ( w1.NeedGradient )
                    {
                        w1.AddMulGradient( w2.TWeight, wt.TGradient );
                    }
                    if ( w2.NeedGradient )
                    {
                        w2.AddMulGradient( w1.TWeight, wt.TGradient );
                    }
                    if ( w3.NeedGradient )
                    {
                        w3.AddMulGradient( w4.TWeight, wt.TGradient );
                    }
                    if ( w4.NeedGradient )
                    {
                        w4.AddMulGradient( w3.TWeight, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                // These tensors' weights will be used during back-propogation, so we unbind them from the computing graph
                w1.UnbindFromComputeGraph();
                w2.UnbindFromComputeGraph();
                w3.UnbindFromComputeGraph();
                w4.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor EltMul( WeightTensor w1, WeightTensor w2 )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w1.Sizes, _DeviceId, name: $"{GetHashString( w1.Name, w2.Name )}.EltMul", graphToBind: this, needGradient: (w1.NeedGradient || w2.NeedGradient) );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w1, w2 }, wt );
#endif
            Ops.Mul( wt.TWeight, w1.TWeight, w2.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    wt.ReleaseWeight();
                    if ( w1.NeedGradient )
                    {
                        w1.AddMulGradient( w2.TWeight, wt.TGradient );
                    }
                    if ( w2.NeedGradient )
                    {
                        w2.AddMulGradient( w1.TWeight, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                w1.UnbindFromComputeGraph();
                w2.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor Add( WeightTensor w1, WeightTensor w2, bool inPlace = false )
        {
            WeightTensor wt;
            if ( inPlace )
            {
                wt = w1.CopyWeightsRef( $"{GetHashString( w1.Name )}.Add", needGradient: (w1.NeedGradient || w2.NeedGradient) );
            }
            else
            {
                wt = _WeightTensorFactory.CreateWeightTensor( w1.Sizes, _DeviceId, name: $"{GetHashString( w1.Name, w2.Name )}.Add", graphToBind: this, needGradient: (w1.NeedGradient || w2.NeedGradient) );
            }
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w1, w2 }, wt );
#endif
            Ops.Add( wt.TWeight, w1.TWeight, w2.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    wt.ReleaseWeight();
                    if ( w1.NeedGradient )
                    {
                        if ( wt.TGradient.IsOwnerExclusive() && w1.IsGradientNull() )
                        {
                            w1.TGradient = wt.TGradient.CopyRef();
                        }
                        else
                        {
                            w1.CopyOrAddGradient( wt );
                        }
                    }
                    if ( w2.NeedGradient )
                    {
                        if ( wt.TGradient.IsOwnerExclusive() && w2.IsGradientNull() )
                        {
                            w2.TGradient = wt.TGradient.CopyRef();
                        }
                        else
                        {
                            w2.CopyOrAddGradient( wt );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Sum( WeightTensor w, int dim )
        {
            var newSizes = w.Sizes.ToArray();
                newSizes[ dim ] = 1;

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( newSizes, _DeviceId, name: $"{w.Name}.Sum", graphToBind: this, needGradient: w.NeedGradient );
            Ops.Sum( wt.TWeight, w.TWeight, dim );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        using var tmp = wt.TGradient.Expand( w.Sizes );
                        w.CopyOrAddGradient( tmp );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Log( WeightTensor w )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Log", graphToBind: this, needGradient: w.NeedGradient );

            Ops.Log( wt.TWeight, w.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        Ops.AddDiv( w.TGradient, w.TGradient, wt.TGradient, w.TWeight );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                w.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor Add( WeightTensor w, float v )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.AddTV", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w }, wt );
#endif
            Ops.Add( wt.TWeight, w.TWeight, v );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        if ( wt.TGradient.IsOwnerExclusive() && w.IsGradientNull() )
                        {
                            w.TGradient = wt.TGradient.CopyRef();
                        }
                        else
                        {
                            w.CopyOrAddGradient( wt );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Sub( float v, WeightTensor w )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.SubVT", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w }, wt );
#endif
            Ops.Sub( wt.TWeight, v, w.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        Ops.Sub( w.TGradient, w.TGradient, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Tanh( WeightTensor w )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Tanh", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Tanh( wt.TWeight, w.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        w.AddTanhGradient( wt );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Relu( WeightTensor w, bool inPlace = false )
        {
            WeightTensor wt;
            if ( inPlace )
            {
                wt = w.CopyWeightsRef( $"{GetHashString( w.Name )}.Relu", needGradient: w.NeedGradient );
            }
            else
            {
                wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Relu", graphToBind: this, needGradient: w.NeedGradient );
            }
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Relu( wt.TWeight, w.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        if ( inPlace && w.IsGradientNull() && wt.TGradient.IsOwnerExclusive() )
                        {
                            w.TGradient = wt.TGradient.CopyRef();
                            Ops.ReluD( w.TGradient, w.TWeight, w.TGradient );
                        }
                        else
                        {
                            Ops.AddReluD( w.TGradient, w.TGradient, w.TWeight, wt.TGradient );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                w.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor MulBatch( WeightTensor w1, WeightTensor w2, float alpha = 1.0f )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { w1.TWeight.Sizes[ 0 ], w1.TWeight.Sizes[ 1 ], w2.TWeight.Sizes[ 2 ] }, _DeviceId, name: $"{GetHashString( w1.Name, w2.Name )}.MulBatch", graphToBind: this, needGradient: (w1.NeedGradient || w2.NeedGradient) );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w1, w2 }, wt );
#endif
            Ops.AddmmBatch( wt.TWeight, 0.0f, wt.TWeight, alpha, w1.TWeight, w2.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    wt.ReleaseWeight();
                    if ( w1.NeedGradient )
                    {
                        using Tensor tW2 = w2.TWeight.Transpose( 1, 2 );
                        Ops.AddmmBatch( w1.TGradient, 1.0f, w1.TGradient, alpha, wt.TGradient, tW2 );
                    }
                    if ( w2.NeedGradient )
                    {
                        using Tensor tW1 = w1.TWeight.Transpose( 1, 2 );
                        Ops.AddmmBatch( w2.TGradient, 1.0f, w2.TGradient, alpha, tW1, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                w1.UnbindFromComputeGraph();
                w2.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor Mul( WeightTensor w1, WeightTensor w2, float alpha = 1.0f )
        {
            int n = w1.Rows;
            int d = w2.Columns;
            WeightTensor wt;

            wt = _WeightTensorFactory.CreateWeightTensor( n, d, _DeviceId, name: $"{GetHashString( w1.Name, w2.Name )}.Mul", graphToBind: this, needGradient: (w1.NeedGradient || w2.NeedGradient) );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { w1, w2 }, wt );
#endif
            Ops.Addmm( wt.TWeight, 0.0f, wt.TWeight, alpha, w1.TWeight, w2.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    wt.ReleaseWeight();
                    if ( w1.NeedGradient )
                    {
                        using Tensor tW2 = w2.TWeight.Transpose();
                        Ops.Addmm( w1.TGradient, 1.0f, w1.TGradient, alpha, wt.TGradient, tW2 );
                    }
                    if ( w2.NeedGradient )
                    {
                        using Tensor tW1 = w1.TWeight.Transpose();
                        Ops.Addmm( w2.TGradient, 1.0f, w2.TGradient, alpha, tW1, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                w1.UnbindFromComputeGraph();
                w2.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor Affine( WeightTensor m1, WeightTensor m2, WeightTensor mbias, float alpha = 1.0f )
        {
            if ( m1 == null ) throw (new ArgumentNullException( $"m1 tensor is null" ));
            if ( m2 == null ) throw (new ArgumentNullException( $"m2 tensor is null" ));
            if ( mbias == null ) throw (new ArgumentNullException( $"mbias tensor is null" ));

            WeightTensor t1 = m1;
            WeightTensor t2 = m2;
            WeightTensor t3 = mbias;

            int n = t1.Rows;
            int d = t2.Columns;
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( n, d, _DeviceId, name: $"{GetHashString( m1.Name, m2.Name, mbias.Name )}.Affine", graphToBind: this, needGradient: (t1.NeedGradient || t2.NeedGradient || t3.NeedGradient) );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { m1, m2, mbias }, wt );
#endif
            using ( var t3WExp = t3.TWeight.Expand( n, d ) )
            {
                Ops.Addmm( wt.TWeight, 1.0f, t3WExp, alpha, t1.TWeight, t2.TWeight );
            }

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    wt.ReleaseWeight();
                    if ( t3.NeedGradient )
                    {
                        using Tensor t3G = t3.TGradient.Expand( n, d );
                        Ops.Add( t3G, t3G, wt.TGradient );
                    }
                    if ( t2.NeedGradient )
                    {
                        using Tensor tW2 = t2.TWeight.Transpose();
                        Ops.Addmm( t1.TGradient, 1.0f, t1.TGradient, alpha, wt.TGradient, tW2 );
                    }
                    if ( t1.NeedGradient )
                    {
                        using Tensor tW1 = t1.TWeight.Transpose();
                        Ops.Addmm( t2.TGradient, 1.0f, t2.TGradient, alpha, tW1, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                t1.UnbindFromComputeGraph();
                t2.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor Transpose( WeightTensor w, int dim1, int dim2 )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Transpose", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            wt.TWeight = w.TWeight.Transpose( dim1, dim2 );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        bool isOwnerExclusive = wt.TGradient.IsOwnerExclusive();
                        using Tensor gT = wt.TGradient.Transpose( dim1, dim2 );
                        if ( isOwnerExclusive && w.IsGradientNull() )
                        {
                            w.TGradient = gT.CopyRef();
                        }
                        else
                        {
                            w.CopyOrAddGradient( gT, wt.Name );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Transpose( WeightTensor w )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Columns, w.Rows, _DeviceId, name: $"{GetHashString( w.Name )}.Transpose", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            wt.TWeight = w.TWeight.Transpose();
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        bool isOwnerExclusive = wt.TGradient.IsOwnerExclusive();

                        using Tensor gT = wt.TGradient.Transpose();
                        if ( isOwnerExclusive && w.IsGradientNull() )
                        {
                            w.TGradient = gT.CopyRef();
                        }
                        else
                        {
                            w.CopyOrAddGradient( gT, wt.Name );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Argmax( WeightTensor w, int dim )
        {
            Tensor argMaxT = Ops.Argmax( null, w.TWeight, dim );

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( argMaxT.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Argmax", graphToBind: this, needGradient: w.NeedGradient );
            wt.TWeight = argMaxT;

            if ( _NeedsBackProp )
            {
                throw new NotSupportedException( $"Argmax operation doesn't support back propagation." );
            }

            return (wt);
        }

        private static double PowerA( double a, double b )
        {
            int tmp  = (int) (BitConverter.DoubleToInt64Bits( a ) >> 32);
            int tmp2 = (int) (b * (tmp - 1072632447) + 1072632447);
            return BitConverter.Int64BitsToDouble( ((long) tmp2) << 32 );
        }

        private static double Exp( double x )
        {
            var tmp = (long) (1512775 * x + 1072632447);
            return BitConverter.Int64BitsToDouble( tmp << 32 );
        }

        /// <summary>
        /// Top-P sampling for each row in given tensor
        /// </summary>
        /// <returns>The sampled index</returns>
        public WeightTensor TopPSampleIndice( WeightTensor w, List<List<int>> seqs, float topP = 0.95f, float repeatPenalty = 2.0f, float distancePenalty = 10.0f )
        {
            float[] weights = w.ToWeightArray();
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { w.Rows, 1 }, _DeviceId, name: $"{GetHashString( w.Name )}.Sample", graphToBind: this, needGradient: w.NeedGradient );

            //var locker  = new object();
            var rnd     = new Random( DateTime.Now.Millisecond );
            var indices = new float[ w.Rows ];
            float thresholdValue = 1.0f / (float) (w.Columns * 10000.0);

            var tokenId2OffsetInSeq = new Dictionary<int, int>(); // <tokenId, offsetInSeq>. The last offset of the token in the given sequence
            var tokenId2Cnt         = new Dictionary<int, int>(); // <tokenId, count> The number of token in the given sequences
            var weight2tokenId      = new SortedDictionary<float, int>();

            for ( int i = 0; i < w.Rows; i++ )
            {
                int offset = i * w.Columns;
                List<int> seq = seqs[ i ];

                tokenId2OffsetInSeq.Clear();
                tokenId2Cnt        .Clear();
                for ( int j = 0; j < seq.Count; j++ )
                {
                    var seq_j = seq[ j ];
                    if ( !tokenId2OffsetInSeq.ContainsKey( seq_j ) )
                    {
                        tokenId2OffsetInSeq.Add( seq_j, j );
                        tokenId2Cnt.Add( seq_j, 0 );
                    }
                    else
                    {
                        tokenId2OffsetInSeq[ seq_j ] = j;
                    }
                    tokenId2Cnt[ seq_j ]++;
                }

                if ( topP == 0.0f )
                {
                    var maxWeight = float.MinValue;
                    int maxWeightIndice = -1;

                    for ( int j = 0; j < w.Columns; j++ )
                    {
                        float weight = weights[ offset + j ];
                        if ( Math.Abs( weight ) < thresholdValue )
                        {
                            continue;
                        }

                        //Decay weights if tokens has already been generated before
                        if ( tokenId2OffsetInSeq.TryGetValue( j, out var offsetInSeq ) )
                        {
                            weight = (float) ((weight * (1.0 - Exp( (offsetInSeq + 1 - seq.Count) / distancePenalty ))) / PowerA( repeatPenalty, tokenId2Cnt[ j ] ));
                        }

                        if ( Math.Abs( weight ) < thresholdValue )
                        {
                            continue;
                        }

                        if ( maxWeight < weight )
                        {
                            maxWeight = weight;
                            maxWeightIndice = j;
                        }
                    }

                    indices[ i ] = maxWeightIndice;
                }
                else
                {
                    weight2tokenId.Clear();
                    float adjustedSum = 0.0f;
                    for ( int j = 0; j < w.Columns; j++ )
                    {
                        float weight = weights[ offset + j ];
                        if ( (Math.Abs( weight ) < thresholdValue) || weight2tokenId.ContainsKey( weight ) )
                        {
                            continue;
                        }

                        //Decay weights if tokens has already been generated before
                        if ( tokenId2OffsetInSeq.TryGetValue( j, out var offsetInSeq ) )
                        {
                            weight = (float) ((weight * (1.0 - Exp( (offsetInSeq + 1 - seq.Count) / distancePenalty ))) / PowerA( repeatPenalty, tokenId2Cnt[ j ] ));
                        }

                        if ( (Math.Abs( weight ) < thresholdValue) || weight2tokenId.ContainsKey( weight ) )
                        {
                            continue;
                        }

                        adjustedSum += weight;
                        weight2tokenId.Add( weight, j );
                    }

                    float acc = 0.0f;
                    //float seed = 0.0f;
                    //lock ( locker )
                    //{
                    var seed = (float) rnd.NextDouble() * topP * adjustedSum;
                    //}

                    foreach ( var pair in weight2tokenId.Reverse() )
                    {
                        acc += pair.Key;

                        if ( seed <= acc )
                        {
                            indices[ i ] = pair.Value;
                            break;
                        }
                    }
                }
            }

            wt.SetWeightArray( indices );

            if ( _NeedsBackProp )
            {
                throw (new NotSupportedException( $"TopPSampleIndice operation doesn't support back propagation." ));
            }

            return (wt);
        }

        public WeightTensor Max( WeightTensor w, int dim )
        {
            Tensor argMaxT = Ops.Max( null, w.TWeight, dim );

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( argMaxT.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Max", graphToBind: this, needGradient: w.NeedGradient );
            wt.TWeight = argMaxT;

            if ( _NeedsBackProp )
            {
                throw (new NotSupportedException( $"Argmax operation doesn't support back propagation." ));
            }

            return (wt);
        }

        public WeightTensor Softmax( WeightTensor w, bool inPlace, bool runGradients = true )
        {
            WeightTensor wt;
            if ( inPlace )
            {
                wt = w.CopyWeightsRef( $"{GetHashString( w.Name )}.Softmax", needGradient: runGradients && w.NeedGradient );
            }
            else
            {
                wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Softmax", graphToBind: this, needGradient: runGradients && w.NeedGradient );
            }
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Softmax( wt.TWeight, w.TWeight );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( runGradients && w.NeedGradient )
                    {
                        if ( inPlace && w.IsGradientNull() && wt.TGradient.IsOwnerExclusive() )
                        {
                            w.TGradient = wt.TGradient.CopyRef();
                        }
                        w.AddSoftmaxGradient( wt, inPlace );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                wt.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor Peek( WeightTensor w, int dim, int ix, int num = 1 )
        {
            long[] sizes = w.Sizes.ToArray();
                   sizes[ dim ] = num;

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Peek", graphToBind: this, needGradient: w.NeedGradient );
            wt.TWeight   = w.TWeight.Narrow( dim, ix, num );
            wt.TGradient = wt.NeedGradient ? w.TGradient.Narrow( dim, ix, num ) : null;
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            if ( _NeedsBackProp )
            {
                void backward() => wt.Dispose();
                _BackProp.Add( backward );
            }

            return (wt);
        }

        private string GetHashString( params string[] inputStrings )
        {
            //if (_VisualizeNetwork)
            //{
            //    string inputString = string.Join("_", inputStrings);
            //    StringBuilder sb = new StringBuilder();
            //    foreach (byte b in GetHash(inputString))
            //    {
            //        sb.Append(b.ToString("X2"));
            //    }

            //    return sb.ToString();
            //}
            return (string.Empty);
        }

        public WeightTensor IndexSelect( WeightTensor src, float[] idxs )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { idxs.Length, src.Sizes[ ^1 ] }, _DeviceId, name: $"{GetHashString( src.Name )}.IndexSelect", graphToBind: this, needGradient: src.NeedGradient );

            var indice = new Tensor( src.Allocator, DType.Float32, idxs.Length );
            indice.CopyFrom( idxs );
            Ops.IndexSelect( wt.TWeight, src.TWeight, indice );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( src.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        Ops.IndexSelectGrad( src.TGradient, wt.TGradient, indice );
                    }
                    wt.Dispose();
                    indice.Dispose();
                };
                _BackProp.Add( backward );
            }
            else
            {
                indice.Dispose();
            }

            return (wt);
        }

        public WeightTensor Concate( int dim, params WeightTensor[] wl ) => Concate( wl.ToList(), dim );
        public WeightTensor Concate( List< WeightTensor > wl, int dim )
        {
            if ( wl.Count == 1 )
            {
                return wl[ 0 ];
            }

            var wlNameList = new List<string>();
            var twl = new List<Tensor>();
            long sumDimSize = 0;
            var needGradient = false;

            foreach ( WeightTensor item in wl )
            {
                WeightTensor m = item;
                sumDimSize += m.Sizes[ dim ];

                twl.Add( m.TWeight );
                wlNameList.Add( item.Name );

                needGradient = (needGradient || m.NeedGradient);
            }

            var newSizes = new long[ wl[ 0 ].Sizes.Length ];
            for ( int i = 0; i < newSizes.Length; i++ )
            {
                newSizes[ i ] = wl[ 0 ].Sizes[ i ];
            }
            newSizes[ dim ] = sumDimSize;

            var wlName = string.Join( "_", wlNameList );
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( newSizes, _DeviceId, name: $"{GetHashString( wlName )}.Concat", graphToBind: this, needGradient: needGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Concat( wt.TWeight, dim, twl.ToArray() );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    wt.ReleaseWeight();
                    var isOwnerExclusive = wt.TGradient.IsOwnerExclusive();

                    long sx = 0;
                    foreach ( WeightTensor item in wl )
                    {
                        WeightTensor m = item;
                        if ( item.NeedGradient )
                        {
                            using Tensor tTmp = wt.TGradient.Narrow( dim, sx, m.Sizes[ dim ] );
                            if ( isOwnerExclusive && m.IsGradientNull() )
                            {
                                m.TGradient = tTmp.CopyRef();
                            }
                            else
                            {
                                m.CopyOrAddGradient( tTmp, wt.Name );
                            }
                        }

                        sx += m.Sizes[ dim ];
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor TransposeBatch( WeightTensor w, int batchSize )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.TransposeBatch", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            int sizeEveryBatch = w.Rows / batchSize;
            using ( Tensor tWView = w.TWeight.View( sizeEveryBatch, batchSize, w.Columns ) )
            {
                using Tensor tWViewPermute = tWView.Permute( 1, 0, 2 );
                using Tensor tW2 = Ops.AsContiguous( tWViewPermute );
                wt.TWeight = tW2.View( w.Rows, w.Columns );
            }

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        using Tensor g = w.TGradient.View( sizeEveryBatch, batchSize, w.Columns );
                        using Tensor t2 = wt.TGradient.View( batchSize, sizeEveryBatch, w.Columns );
                        using Tensor t2Permute = t2.Permute( 1, 0, 2 );
                        Ops.Add( g, g, t2Permute );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public List<WeightTensor> SplitColumns2( WeightTensor w, params int[] sizes )
        {
            var resList = new List<WeightTensor>();

            int x = 0;
            foreach ( int size in sizes )
            {
                WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Rows, size, _DeviceId, name: $"{GetHashString( w.Name )}.SplitColumn", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
                VisualizeNodes( w, wt );
#endif
                wt.TWeight = w.TWeight.Narrow( 1, x, size );
                resList.Add( wt );

                x += size;
            }

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        x = 0;
                        int i = 0;
                        foreach ( WeightTensor item in resList )
                        {
                            WeightTensor item_i = item;
                            using ( Tensor mG = w.TGradient.Narrow( 1, x, sizes[ i ] ) )
                            {
                                Ops.Add( mG, mG, item_i.TGradient );
                            }

                            item.Dispose();

                            x += sizes[ i ];
                            i++;
                        }
                    }
                    else
                    {
                        foreach ( WeightTensor item in resList )
                        {
                            item.Dispose();
                        }
                    }
                };
                _BackProp.Add( backward );
            }

            return (resList);
        }

        public WeightTensor AsContiguous( WeightTensor w, bool shareTensor = true )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.AsContiguous", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            wt.TWeight = Ops.AsContiguous( w.TWeight );

            if ( shareTensor )
            {
                w.ReleaseWeight();
                w.TWeight = wt.TWeight.CopyRef();
            }

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        if ( wt.TGradient.IsOwnerExclusive() && w.IsGradientNull() )
                        {
                            w.TGradient = wt.TGradient.CopyRef();
                        }
                        else
                        {
                            w.CopyOrAddGradient( wt );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor View( WeightTensor w, params long[] dims )
        {
            var hasNegOne = false;
            int negOneIdx = 0;
            long totalGivenSize = 1;
            for ( int i = 0; i < dims.Length; i++ )
            {
                long dim = dims[ i ];
                if ( dim == -1 )
                {
                    if ( hasNegOne )
                    {
                        throw (new ArgumentException( $"View operation only allows single -1 in dims." ));
                    }

                    hasNegOne = true;
                    negOneIdx = i;
                }
                else
                {
                    totalGivenSize *= dim;
                }
            }

            if ( hasNegOne )
            {
                long totalSrcSize = 1;
                foreach ( int size in w.Sizes )
                {
                    totalSrcSize *= size;
                }

                dims[ negOneIdx ] = totalSrcSize / totalGivenSize;
            }

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( dims, _DeviceId, name: $"{w.Name}.View", graphToBind: this, needGradient: w.NeedGradient );
            //VisualizeNodes( w, wt );

            wt.TWeight = w.TWeight.View( dims );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        bool isOwnerExclusive = wt.TGradient.IsOwnerExclusive();

                        using Tensor resGConti = Ops.AsContiguous( wt.TGradient );
                        using Tensor resG = resGConti.View( w.Sizes );
                        if ( isOwnerExclusive && w.IsGradientNull() )
                        {
                            w.TGradient = resG.CopyRef();
                        }
                        else
                        {
                            w.CopyOrAddGradient( resG, wt.Name );
                        }
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Scatter( WeightTensor src, WeightTensor indices, int dim, params long[] shape )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( shape, _DeviceId, name: $"{GetHashString( src.Name + indices.Name )}.Scatter", graphToBind: this, needGradient: src.NeedGradient );

            Ops.Fill( wt.TWeight, 0.0f );
            Ops.Scatter( wt.TWeight, src.TWeight, dim, indices.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( src.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        using var tmp = Ops.Gather( null, wt.TGradient, dim, indices.TWeight );
                        src.CopyOrAddGradient( tmp );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor ScatterAdd( WeightTensor src, WeightTensor indices, int dim, params long[] shape )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( shape, _DeviceId, name: $"{GetHashString( src.Name + indices.Name )}.Scatter", graphToBind: this, needGradient: src.NeedGradient );

            Ops.Fill( wt.TWeight, 0.0f );
            Ops.ScatterAdd( wt.TWeight, src.TWeight, dim, indices.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( src.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        using var tmp = Ops.Gather( null, wt.TGradient, dim, indices.TWeight );
                        src.CopyOrAddGradient( tmp );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Scatter( WeightTensor indices, float val, int dim, bool needGradient = true, params long[] shape )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( shape, _DeviceId, name: $"{GetHashString( indices.Name )}.Scatter", graphToBind: this, needGradient: needGradient );

            Ops.Fill( wt.TWeight, 0.0f );
            Ops.ScatterFill( wt.TWeight, val, dim, indices.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( needGradient )
                    {
                        wt.ReleaseWeight();
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Expand( WeightTensor w, params long[] dims )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( dims, _DeviceId, name: $"{GetHashString( w.Name )}.Expand", graphToBind: this, needGradient: w.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            wt.TWeight = w.TWeight.Expand( dims );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( w.NeedGradient )
                    {
                        wt.ReleaseWeight();

                        using var tmpMGrad = w.TGradient.Expand( dims ); // expand input tensor at first
                        Ops.AtomicAdd( tmpMGrad, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public (WeightTensor r1, WeightTensor r2) SplitColumns( WeightTensor w, int size1, int size2 )
        {
            List<WeightTensor> res = SplitColumns2( w, size1, size2 );
            return (res[ 0 ], res[ 1 ]);
        }

        public (WeightTensor r1, WeightTensor r2, WeightTensor r3) SplitColumns( WeightTensor w, int size1, int size2, int size3 )
        {
            List<WeightTensor> res = SplitColumns2( w, size1, size2, size3 );
            return (res[ 0 ], res[ 1 ], res[ 2 ]);
        }

        private Tensor BuildRandomTensor( int rows, int columns, int batchSize, float prob )
        {
            using var noise = new Tensor( TensorAllocator.Allocator( _DeviceId ), DType.Float32, rows / batchSize, columns );
            float[] w = Tensors.RandomGenerator.BuildRandomBernoulliWeight( new long[] { rows / batchSize, columns }, prob );
            noise.SetElementsAsFloat( w );

            if ( (rows / batchSize) == 1 )
            {
                return noise.Expand( rows, columns );
            }
            else
            {
                return noise.RepeatTensor( batchSize, 1 );
            }
        }

        public WeightTensor LayerNorm( WeightTensor src, WeightTensor alpha, WeightTensor beta, float eps = 1e-9f )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( src.Sizes, _DeviceId, name: $"{GetHashString( src.Name, alpha.Name, beta.Name )}.LayerNorm", graphToBind: this, needGradient: src.NeedGradient );
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( new WeightTensor[] { src, alpha, beta }, wt );
#endif
            Ops.LayerNorm( wt.TWeight, src.TWeight, alpha.TWeight, beta.TWeight, eps );
            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( src.NeedGradient )
                    {
                        Ops.LayerNormGrad( src.TGradient, alpha.TGradient, beta.TGradient, wt.TGradient, wt.TWeight, src.TWeight, alpha.TWeight, beta.TWeight, eps );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );

                src.UnbindFromComputeGraph();
                alpha.UnbindFromComputeGraph();
                beta.UnbindFromComputeGraph();
            }

            return (wt);
        }

        public WeightTensor Dropout( WeightTensor w, int batchSize, float drop_prob, bool inPlace = false )
        {
            if ( drop_prob == 0 || !_NeedsBackProp )
            {
                return (w);
            }

            // Generate noise tensor
            float p = 1.0f - drop_prob;
            Tensor noise = BuildRandomTensor( w.Rows, w.Columns, batchSize, p );

            WeightTensor wt;
            if ( inPlace )
            {
                wt = w.CopyWeightsRef( $"{GetHashString( w.Name )}.Dropout", needGradient: w.NeedGradient );
            }
            else
            {
                wt = _WeightTensorFactory.CreateWeightTensor( w.Sizes, _DeviceId, name: $"{GetHashString( w.Name )}.Dropout", graphToBind: this, needGradient: w.NeedGradient );
            }
#if USE_VISUALIZE_NETWORK
            VisualizeNodes( w, wt );
#endif
            Ops.Mul( wt.TWeight, w.TWeight, noise );

            void backward()
            {
                if ( w.NeedGradient )
                {
                    wt.ReleaseWeight();

                    if ( inPlace && w.IsGradientNull() && wt.TGradient.IsOwnerExclusive() )
                    {
                        w.TGradient = wt.TGradient.CopyRef();
                    }

                    w.AddMulGradient( noise, wt.TGradient, inPlace );
                }
                wt.Dispose();
                noise.Dispose();
            };
            _BackProp.Add( backward );

            return (wt);
        }

        public WeightTensor Gather( WeightTensor src, WeightTensor indices, int dim )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( indices.Sizes, _DeviceId, name: $"Gather_{_DeviceId}", graphToBind: this, needGradient: src.NeedGradient );
            Ops.Gather( wt.TWeight, src.TWeight, dim, indices.TWeight );

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( src.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        Ops.ScatterAdd( src.TGradient, wt.TGradient, dim, indices.TWeight );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor Select( WeightTensor src, int dim, int index )
        {
            var resTWeight = src.TWeight.Select( dim, index );

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( resTWeight.Sizes, _DeviceId, name: $"Select_{_DeviceId}", graphToBind: this, needGradient: src.NeedGradient );
            wt.TWeight = resTWeight;

            if ( _NeedsBackProp )
            {
                void backward()
                {
                    if ( src.NeedGradient )
                    {
                        wt.ReleaseWeight();
                        using var tmpG = src.TGradient.Select( dim, index );
                        Ops.Add( tmpG, tmpG, wt.TGradient );
                    }
                    wt.Dispose();
                };
                _BackProp.Add( backward );
            }

            return (wt);
        }

        /// <returns>Shape: [batch_size, seq_len]</returns>
        public WeightTensor LeftShiftTokens( List<List<int>> input, int lastTokenToPad )
        {
            var buf = new float[ input.Count * input[ 0 ].Count ];
            for ( int i = 0; i < input.Count; i++ )
            {
                var input_i = input[ i ];
                var offset = i * input_i.Count;
                for ( int j = 0; j < input_i.Count - 1; j++ )
                {
                    buf[ offset + j ] = input_i[ j + 1 ];
                }
                buf[ (i + 1) * input_i.Count - 1 ] = lastTokenToPad;
            }

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( input.Count, input[ 0 ].Count, _DeviceId, name: $"LeftShiftTokens_{_DeviceId}", graphToBind: this, needGradient: false );
            wt.SetWeightArray( buf );

            if ( _NeedsBackProp )
            {
                void backward() => wt.Dispose();
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor CreateTokensTensor( List<List<int>> input )
        {
            var buf = new float[ input.Count * input[ 0 ].Count ];
            for ( int i = 0; i < input.Count; i++ )
            {
                var input_i = input[ i ];
                var offset = i * input_i.Count;
                for ( int j = 0; j < input_i.Count; j++ )
                {
                    buf[ offset + j ] = input_i[ j ];
                }
            }

            WeightTensor res = _WeightTensorFactory.CreateWeightTensor( input.Count, input[ 0 ].Count, _DeviceId, name: $"TokensTensor_{_DeviceId}", graphToBind: this, needGradient: false );
            res.SetWeightArray( buf );

            if ( _NeedsBackProp )
            {
                void backward() => res.Dispose();
                _BackProp.Add( backward );
            }

            return (res);
        }

        /// <returns>shape: (batch_size, sequence_padded_length, dim)</returns>
        public WeightTensor BuildFeatureMask( int paddedLength, List<int> appliedLengths, int dim )
        {
            var buf = new float[ appliedLengths.Count * paddedLength * dim ];
            //Array.Fill( buf, 0.0f );
            for ( int k = 0; k < appliedLengths.Count; k++ )
            {
                var appliedLengths_k = appliedLengths[ k ];
                var offset = k * (paddedLength * dim);
                for ( int i = 0; i < appliedLengths_k; i++ )
                {
                    Array.Fill( buf, 1.0f, offset + i * dim, dim );
                }
            }

            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { appliedLengths.Count, paddedLength, dim }, _DeviceId, name: $"FeatureMask_{_DeviceId}", graphToBind: this, needGradient: false );
            wt.SetWeightArray( buf );

            if ( _NeedsBackProp )
            {
                void backward() => wt.Dispose();
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor BuildPadSelfMask( int paddedLength, float[] originalLengths )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { originalLengths.Length, paddedLength, paddedLength }, _DeviceId, name: $"SelfMask_{_DeviceId}", graphToBind: this, needGradient: false );
            using ( var originalLengthsTensor = new Tensor( wt.Allocator, DType.Float32, originalLengths.Length ) )
            {
                originalLengthsTensor.CopyFrom( originalLengths );
                Ops.BuildSelfMask( wt.TWeight, originalLengthsTensor, paddedLength, 0.0f, -99999999.0f );
            }

            if ( _NeedsBackProp )
            {
                void backward() => wt.Dispose();
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor BuildSelfTriMask( int paddedLength, float[] originalLengths )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { originalLengths.Length, paddedLength, paddedLength }, _DeviceId, name: $"SelfTriMask_{_DeviceId}", graphToBind: this, needGradient: false );
            using ( var originalLengthsTensor = new Tensor( wt.Allocator, DType.Float32, originalLengths.Length ) )
            {
                originalLengthsTensor.CopyFrom( originalLengths );
                Ops.BuildSelfTriMask( wt.TWeight, originalLengthsTensor, paddedLength, 0.0f, -99999999.0f );
            }

            if ( _NeedsBackProp )
            {
                void backward() => wt.Dispose();
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor BuildTriMask( int paddedLength, int batchSize )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { paddedLength, paddedLength }, _DeviceId, name: $"SelfTriMask2_{_DeviceId}", graphToBind: this, needGradient: false );
            Ops.BuildTriMask( wt.TWeight, 0.0f, -99999999.0f );

            if ( _NeedsBackProp )
            {
                void backward() => wt.Dispose();
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public WeightTensor BuildSrcTgtMask( int srcPaddedLength, int tgtPaddedLength, float[] tgtOriginalLengths, float[] srcOriginalLengths )
        {
            WeightTensor wt = _WeightTensorFactory.CreateWeightTensor( new long[] { tgtOriginalLengths.Length, tgtPaddedLength, srcPaddedLength }, _DeviceId, name: $"SrcTgtMask_{_DeviceId}", graphToBind: this, needGradient: false );

            using ( var tgtOriginalLengthsTensor = new Tensor( wt.Allocator, DType.Float32, tgtOriginalLengths.Length ) )            
            using ( var srcOriginalLengthsTensor = new Tensor( wt.Allocator, DType.Float32, srcOriginalLengths.Length ) )
            {
                srcOriginalLengthsTensor.CopyFrom( srcOriginalLengths );
                tgtOriginalLengthsTensor.CopyFrom( tgtOriginalLengths );
                Ops.BuildSrcTgtMask( wt.TWeight, srcOriginalLengthsTensor, tgtOriginalLengthsTensor, srcPaddedLength, tgtPaddedLength, 0.0f, -99999999.0f );
            }
            
            if ( _NeedsBackProp )
            {
                void backward() => wt.Dispose();
                _BackProp.Add( backward );
            }

            return (wt);
        }

        public float CrossEntropyLoss( WeightTensor probs, WeightTensor truthTgtSeqs, float graident = 1.0f, float smooth = 0.0f )
        {
            var scatterIdxTensor = View( truthTgtSeqs, new long[] { -1, 1 } );
            var loss = Gather( probs, scatterIdxTensor, 1 );

            if ( smooth > 0.0f )
            {
                loss = Add( loss, smooth );
            }

            loss = Log( loss );
            loss = Mul( loss, -1.0f );
            loss.FillGradient( graident );

            return (loss.ToWeightArray().Sum() / loss.ElementCount);
        }
#if USE_VISUALIZE_NETWORK
        private void VisualizeNodes( WeightTensor sourceNode, WeightTensor targetNode ) => VisualizeNodes( new WeightTensor[] { sourceNode }, targetNode );
        private void VisualizeNodes( IEnumerable< WeightTensor > sourceNodes, WeightTensor targetNode )
        {
            //if ( !_VisualizeNetwork || _DeviceId !=  0)
            //{
            //    return;
            //}

            //// Create node for target tensor
            //int index = targetNode.Name.LastIndexOf('.');
            //Microsoft.Msagl.Drawing.Node tgtNode = _MsGraph.AddNode(targetNode.Name);
            //tgtNode.LabelText = targetNode.Name.Substring(index + 1);

            //if (targetNode.IsTrainable)
            //{
            //    tgtNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;
            //}

            //if (_MsSubGraph != null)
            //{
            //    // Current compute graph is a sub-graph
            //    _MsSubGraph.AddNode(tgtNode);
            //}

            //// Create edges for each source node and target node
            //foreach (WeightTensor sourceNode in sourceNodes)
            //{
            //    if (!sourceNode.Name.IsNullOrEmpty() && !targetNode.Name.IsNullOrEmpty())
            //    {
            //        string key = $"{sourceNode.Name}->{targetNode.Name}";
            //        if (_MsGraph_Edges.Contains(key))
            //        {
            //            continue;
            //        }

            //        int srcIndex = sourceNode.Name.LastIndexOf('.');
            //        Microsoft.Msagl.Drawing.Node srcNode = _MsGraph.AddNode(sourceNode.Name);
            //        srcNode.LabelText = sourceNode.Name.Substring(srcIndex + 1);
            //        if (sourceNode.IsTrainable)
            //        {
            //            srcNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;

            //            if (_MsSubGraph != null)
            //            {
            //                _MsSubGraph.AddNode(srcNode);
            //            }
            //        }

            //        Edge edge = _MsGraph.AddEdge(sourceNode.Name, targetNode.Name);

            //        _MsGraph_Edges.Add(key);
            //    }
            //}
        }
        public void VisualizeNeuralNetToFile( string neuralNetPicFilePath )
        {
            //FastIncrementalLayoutSettings fastSettings = new FastIncrementalLayoutSettings
            //{
            //    AvoidOverlaps = true,
            //    NodeSeparation = 30,
            //    RouteEdges = true
            //};

            //SugiyamaLayoutSettings settings = new SugiyamaLayoutSettings
            //{
            //    FallbackLayoutSettings = fastSettings
            //};

            //_MsGraph.LayoutAlgorithmSettings = settings;

            //Microsoft.Msagl.GraphViewerGdi.GraphRenderer renderer = new Microsoft.Msagl.GraphViewerGdi.GraphRenderer(_MsGraph);
            //renderer.CalculateLayout();

            //System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap((int)_MsGraph.Width, (int)_MsGraph.Height, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
            //renderer.Render(bitmap);

            //bitmap.Save(neuralNetPicFilePath);

            //bitmap.Dispose();
        }
#endif
    }
}
