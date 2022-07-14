using System;
using System.Collections.Generic;
using System.Text;

using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Tensors;
using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    public enum NormType
    {
        None,
        Uniform,
        Normal
    }
    /// <summary>
    /// 
    /// </summary>

    public sealed class WeightTensor : INeuralUnit, IDisposable
    {
        public long[] Sizes { get; set; }
        public int Rows
        {
            get => (int) Sizes[ 0 ];
            set => Sizes[ 0 ] = value;
        }
        public int Columns
        {
            get => (int) Sizes[ 1 ];
            set => Sizes[ 1 ] = value;
        }

        public string Name { get; set; }
        public bool IsTrainable { get; set; }
        public bool NeedGradient { get; set; }
        public int DeviceId { get; set; }
        public float LearningRateFactor { get; set; } = 1.0f;

        private IAllocator _Allocator;
        public IAllocator Allocator => _Allocator;

        private Tensor _TWeight;
        private Tensor _TGradient;
        private static readonly object _Locker = new object();

        private bool _ReleasedWeight;
        private readonly ComputeGraphTensor _ComputeGraphToBind;
        private string _GradientSetName = "None";
        private readonly bool _FanIn;
        private readonly bool _FanOut;
        private readonly NormType _NormType = NormType.None;

        public long ElementCount => _TWeight.ElementCount();
        public Tensor TWeight
        {
            get
            {
                if ( _ReleasedWeight )
                {
                    throw (new Exception( $"The weight '{Name}' has been released, you cannot access it." ));
                }
                if ( _TWeight == null )
                {
                    _TWeight = new Tensor( _Allocator, DType.Float32, Sizes );
                }
                return _TWeight;
            }
            set
            {
                if ( _TWeight != null )
                {
                    throw (new Exception( $"Please call ReleaseWeight function before assign a new value to weight '{Name}'." ));
                }

                _TWeight = value;

                if ( _TWeight != null )
                {
                    Sizes = _TWeight.Sizes;
                    if ( _TGradient != null )
                    {
                        for ( int i = 0; i < Sizes.Length; i++ )
                        {
                            if ( Sizes[ i ] != _TGradient.Sizes[ i ] )
                            {
                                throw new Exception( $"The shape between weights and gradients are different. Name = '{Name}'" );
                            }
                        }
                    }
                    _ReleasedWeight = false;
                }
            }
        }

        public Tensor TGradient
        {
            get
            {
                if ( _TGradient == null )
                {
                    _TGradient = new Tensor( _Allocator, DType.Float32, Sizes );
                    Ops.Fill( _TGradient, 0.0f );

                    _GradientSetName = "Get";
                }
                return _TGradient;
            }

            set
            {
                if ( _TGradient != null )
                {
                    throw new Exception( $"Please call ReleaseGradient function before assign a new value to gradient '{Name}'. This gradient was set by '{_GradientSetName}'" );
                }

                _TGradient = value;

                if ( _TGradient != null )
                {
                    Sizes = _TGradient.Sizes;
                    if ( _TWeight != null )
                    {
                        for ( int i = 0; i < Sizes.Length; i++ )
                        {
                            if ( Sizes[ i ] != _TWeight.Sizes[ i ] )
                            {
                                throw new Exception( $"The shape between weights and gradients are different. Name = '{Name}'" );
                            }
                        }
                    }
                }
            }
        }

        public WeightTensor( long[] sizes, int deviceId, string name = "", bool isTrainable = false, NormType normType = NormType.None, bool fanIn = false, bool fanOut = false, float learningRateFactor = 1.0f, ComputeGraphTensor graphToBind = null, bool needGradient = true )
        {
            Name = name;
            DeviceId = deviceId;
            LearningRateFactor = learningRateFactor;
            IsTrainable = isTrainable;
            NeedGradient = needGradient;
            _Allocator = TensorAllocator.Allocator( DeviceId );
            Sizes = sizes;
            _FanIn = fanIn;
            _FanOut = fanOut;
            _NormType = normType;

            if ( graphToBind != null )
            {
                _ComputeGraphToBind = graphToBind;
                _ComputeGraphToBind.Bind( this );
            }

            if ( normType == NormType.Uniform )
            {
                var scale = (float) Math.Sqrt( 6.0 / (double) (Rows + Columns) );

                if ( fanIn && !fanOut )
                {
                    scale = (float) Math.Sqrt( 3.0 / (double) Rows );
                }
                else if ( !fanIn && fanOut )
                {
                    scale = (float) Math.Sqrt( 3.0 / (double) Columns );
                }

                float[] w = Tensors.RandomGenerator.BuildRandomUniformWeight( Sizes, -scale, scale );
                SetWeightArray( w );
            }
            else if ( normType == NormType.Normal )
            {
                float[] w = Tensors.RandomGenerator.BuildRandomUniformWeight( Sizes, -1.0f, 1.0f );
                SetWeightArray( w );
            }
        }

        public WeightTensor( long[] sizes, float c, int deviceId, string name = "", bool isTrainable = false, float learningRateFactor = 1.0f, bool needGradient = true )
        {
            Name = name;
            DeviceId = deviceId;
            IsTrainable = isTrainable;
            NeedGradient = needGradient;
            LearningRateFactor = learningRateFactor;
            Sizes = sizes;
            _Allocator = TensorAllocator.Allocator( DeviceId );

            var tensor = new Tensor( _Allocator, DType.Float32, Sizes );
            Ops.Fill( tensor, c );

            TWeight = tensor;
        }

        public void UnbindFromComputeGraph()
        {
            if ( _ComputeGraphToBind != null )
            {
                _ComputeGraphToBind.Unbind( this );
            }
        }
        public int GetDeviceId() => DeviceId;
        public INeuralUnit CloneToDeviceAt( int deviceId ) => new WeightTensor( Sizes, deviceId, Name, IsTrainable, normType: _NormType, fanIn: _FanIn, fanOut: _FanOut, needGradient: NeedGradient );

        public void ZeroGradient() => Ops.Fill( TGradient, 0.0f );
        public void CleanWeight() => Ops.Fill( TWeight, 0.0f );
        public void FillGradient( float val ) => Ops.Fill( TGradient, val );
        public float GetWeightAt( long[] indices ) => TWeight.GetElementAsFloat( indices );
        public float GetGradientAt( long[] indices ) => TGradient.GetElementAsFloat( indices );
        public void SetWeightAt( float val, long[] indices ) => TWeight.SetElementAsFloat( val, indices );
        public void CopyWeightsToGradients( WeightTensor src )
        {
            WeightTensor m = src;
            if ( _TGradient != null )
            {
                _TGradient.Dispose();
            }

            _TGradient = m.TWeight.CopyRef();
            _GradientSetName = "CopyWeightsToGradients";
        }

        public void CopyWeightsFrom( WeightTensor src )
        {
            WeightTensor m = src;
            Ops.Copy( TWeight, m.TWeight );
        }

        public void AddGradientFrom( WeightTensor src )
        {
            WeightTensor m = src;
            lock ( _Locker )
            {
                Tensor t = new Tensor( TGradient.Allocator, DType.Float32, Sizes );
                Ops.Copy( t, m.TGradient );
                Ops.Add( TGradient, TGradient, t );

                t.Dispose();
            }
        }
        public float[] ToWeightArray() => TWeight.GetElementsAsFloat( (int) _TWeight.GetStorageSize() );

        private static object _LastCheckDT_lock = new object();
        private static DateTime _LastCheckDT = DateTime.Now;
        public void PrintWeights()
        {
            TimeSpan ts;
            lock ( _LastCheckDT_lock ) ts = (DateTime.Now - _LastCheckDT);
            if ( TimeSpan.FromMinutes( 5.0f ) <= ts )
            {
                lock ( _LastCheckDT_lock ) _LastCheckDT = DateTime.Now;
                var weights = ToWeightArray();

                var sb = new StringBuilder().Append( $"Weights for '{Name}': [" );
                foreach ( var weight in weights )
                {
                    sb.Append( $"{weight:F4}, " );
                }
                sb.Append( ']' );

                Logger.WriteLine( sb.ToString() );
            }
        }

        public void AddSoftmaxGradient( WeightTensor src, bool inPlace = false )
        {
            if ( _TGradient == null )
            {
                _Allocator = TensorAllocator.Allocator( DeviceId );
                _TGradient = new Tensor( _Allocator, DType.Float32, Sizes );
                Ops.SoftmaxGrad( _TGradient, src.TGradient, src.TWeight, false );

                _GradientSetName = "AddSoftmaxGradient";
            }
            else
            {
                Ops.SoftmaxGrad( _TGradient, src.TGradient, src.TWeight, !inPlace );
            }
        }

        public void CopyOrAddGradient( WeightTensor src )
        {
            if ( _TGradient == null )
            {
                _Allocator = TensorAllocator.Allocator( DeviceId );
                _TGradient = new Tensor( _Allocator, DType.Float32, Sizes );
                Ops.Copy( _TGradient, src.TGradient );

                _GradientSetName = "CopyOrAddGradient_WeightTensor";
            }
            else
            {
                Ops.Add( _TGradient, _TGradient, src.TGradient );
            }
        }
        public void Clamp( float min, float max ) => Ops.Clamp( TWeight, TWeight, min, max );

        public void CopyOrAddGradient( Tensor src, string callerName = "" )
        {
            if ( _TGradient == null )
            {
                _Allocator = TensorAllocator.Allocator( DeviceId );
                _TGradient = new Tensor( _Allocator, DType.Float32, Sizes );
                Ops.Copy( _TGradient, src );

                _GradientSetName = $"CopyOrAddGradient_Tensor_CalledBy_{callerName}";
            }
            else
            {
                Ops.Add( _TGradient, _TGradient, src );
            }
        }

        public void AddMulGradient( Tensor w, Tensor g, bool inPlace = false )
        {
            if ( _TGradient == null )
            {
                _Allocator = TensorAllocator.Allocator( DeviceId );
                _TGradient = new Tensor( _Allocator, DType.Float32, Sizes );
                Ops.Mul( _TGradient, w, g );

                _GradientSetName = "AddMulGrdient";
            }
            else
            {
                if ( inPlace )
                {
                    Ops.Mul( _TGradient, w, g );
                }
                else
                {
                    Ops.AddMul( _TGradient, _TGradient, w, g );
                }
            }
        }
        public void AddSigmoidGradient( WeightTensor src )
        {
            if ( _TGradient == null )
            {
                _Allocator = TensorAllocator.Allocator( DeviceId );
                _TGradient = new Tensor( _Allocator, DType.Float32, Sizes );
                Ops.SigmoidD( _TGradient, src.TWeight, src.TGradient );

                _GradientSetName = "AddSigmoidGradient";
            }
            else
            {
                Ops.AddSigmoidD( _TGradient, _TGradient, src.TWeight, src.TGradient );
            }
        }
        public void AddTanhGradient( WeightTensor src )
        {
            if ( _TGradient == null )
            {
                _Allocator = TensorAllocator.Allocator( DeviceId );
                _TGradient = new Tensor( _Allocator, DType.Float32, Sizes );

                Ops.TanhD( _TGradient, src.TWeight, src.TGradient );

                _GradientSetName = "AddTanhGradient";
            }
            else
            {
                Ops.AddTanhD( _TGradient, _TGradient, src.TWeight, src.TGradient );
            }
        }

        //public List<int> GetTopNMaxWeightIdx( int topN )
        //{
        //    var weights = ToWeightArray();
        //    var q = new FixedSizePriorityQueue< ComparableItem< int > >( topN, ComparableItemComparer< int >.Asc );
        //    for ( int i = 0; i < weights.Length; i++ )
        //    {
        //        q.Enqueue( new ComparableItem<int>( weights[ i ], i ) );
        //    }
        //    return q.Select( x => x.Value ).ToList( q.Count );
        //}
        public void SetWeightArray( float[] v ) => TWeight.SetElementsAsFloat( v );
        public WeightTensor CopyWeightsRef( string name, bool needGradient ) => new WeightTensor( Sizes, DeviceId, name, needGradient: needGradient ) { _TWeight = _TWeight.CopyRef() };

        public void Dispose()
        {
            ReleaseWeight();
            ReleaseGradient();
        }

        public bool IsGradientNull() => (_TGradient == null);
        public void ReleaseWeight()
        {
            if ( _TWeight != null )
            {
                _TWeight.Dispose();
                _TWeight = null;
                _ReleasedWeight = true;
            }
        }
        public void ReleaseGradient()
        {
            if ( _TGradient != null )
            {
                _TGradient.Dispose();
                _TGradient = null;
            }
        }

        public void Save( Model model ) => model.AddWeights( Name, ToWeightArray() );
        public void Load( Model model )
        {
            //---Logger.WriteLine( $"Loading weights '{Name}' from the model..." );

            var weights = model.GetWeights( Name );
            if ( weights != null )
            {
                SetWeightArray( weights );
            }
        }
        public List<WeightTensor> GetParams() => new List<WeightTensor>() { this };
    }
}
