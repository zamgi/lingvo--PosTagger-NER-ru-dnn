using System.Collections.Generic;
using System.Threading.Tasks;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal interface IMultiProcessorNetworkWrapper
    {
        void Save( Model model );
        void Load( Model model );
        void SyncWeights();
        void SumGradientsToNetworkOnDefaultDevice();
        INeuralUnit GetNeuralUnitOnDefaultDevice();
        void ZeroGradientsOnAllDevices();
        void ReleaseGradientsOnAllDevices();
    }

    /// <summary>
    /// 
    /// </summary>
    internal sealed class MultiProcessorNetworkWrapper< T > : IMultiProcessorNetworkWrapper where T : INeuralUnit
    {
        private readonly T[] _Networks;
        private readonly int _DefaultDeviceId;
        private readonly T _NetworkOnDefaultDevice;
        private readonly bool _IsStaticWeights;
        private bool _WeightsSynced;

        public MultiProcessorNetworkWrapper( T networkOnDefaultDevice, int[] deviceIds, bool isStaticWeights = false )
        {
            _Networks               = new T[ deviceIds.Length ];
            _DefaultDeviceId        = networkOnDefaultDevice.GetDeviceId();
            _NetworkOnDefaultDevice = networkOnDefaultDevice;
            _IsStaticWeights        = isStaticWeights;
            _WeightsSynced          = false;

            for ( int i = 0; i < deviceIds.Length; i++ )
            {
                var devId = deviceIds[ i ];
                if ( devId == _DefaultDeviceId )
                {
                    _Networks[ i ] = networkOnDefaultDevice;
                }
                else
                {
                    _Networks[ i ] = (T) networkOnDefaultDevice.CloneToDeviceAt( devId );
                }
            }
        }

        /// <summary>
        /// Copy weights from tensors on the default device to all other devices
        /// </summary>
        public void SyncWeights()
        {
            if ( _IsStaticWeights && _WeightsSynced )
            {
                return;
            }

            List< WeightTensor > tensorsOnDefaultDevice = _NetworkOnDefaultDevice.GetParams();
            Parallel.ForEach( _Networks, network =>
            {
                 if ( !network.Equals( _NetworkOnDefaultDevice ) )
                 {
                     List< WeightTensor > tensors = network.GetParams();
                     for ( int j = 0; j < tensors.Count; j++ )
                     {
                         tensors[ j ].CopyWeightsFrom( tensorsOnDefaultDevice[ j ] );
                     }
                 }
            });

            _WeightsSynced = true;
        }

        /// <summary>
        /// Collect gradients from other devices and sum it up to the default device
        /// </summary>
        public void SumGradientsToNetworkOnDefaultDevice()
        {
            if ( _IsStaticWeights )
            {
                return;
            }

            List< WeightTensor > tensorsOnDefaultDevice = _NetworkOnDefaultDevice.GetParams();
            Parallel.ForEach( _Networks, network =>
            {
                 if ( !network.Equals( _NetworkOnDefaultDevice ) )
                 {
                     List< WeightTensor > tensors = network.GetParams();
                     for ( int j = 0; j < tensors.Count; j++ )
                     {
                         tensorsOnDefaultDevice[ j ].AddGradientFrom( tensors[ j ] );
                     }
                 }
            });
        }

        /// <summary>
        /// Fill zero to all gradients on all devices
        /// </summary>
        public void ZeroGradientsOnAllDevices()
        {
            if ( _IsStaticWeights )
            {
                return;
            }

            Parallel.ForEach( _Networks, network =>
            {
                 List< WeightTensor > tensors = network.GetParams();
                 for ( int j = 0; j < tensors.Count; j++ )
                 {
                     tensors[ j ].ZeroGradient();
                 }
            });
        }


        /// <summary>
        /// Release gradients on all devices
        /// </summary>
        public void ReleaseGradientsOnAllDevices()
        {
            if ( _IsStaticWeights )
            {
                return;
            }

            Parallel.ForEach( _Networks, network =>
            {
                 List< WeightTensor > tensors = network.GetParams();
                 for ( int j = 0; j < tensors.Count; j++ )
                 {
                     tensors[ j ].ReleaseGradient();
                 }
            });
        }

        /// <summary>
        /// Save weights of the network on default device to given model
        /// </summary>
        public void Save( Model model )
        {
            if ( !_IsStaticWeights )
            {
                _NetworkOnDefaultDevice.Save( model );
            }
        }

        /// <summary>
        /// Load weights from given model to the network on default device
        /// </summary>
        public void Load( Model model )
        {
            if ( !_IsStaticWeights )
            {
                _NetworkOnDefaultDevice.Load( model );
            }
        }

        public T GetNetworkOnDefaultDevice() => _NetworkOnDefaultDevice;
        public INeuralUnit GetNeuralUnitOnDefaultDevice() => GetNetworkOnDefaultDevice();

        /// <summary>
        /// Return the network on specific device
        /// </summary>
        /// <param name="deviceIdIdx">The device id index. -1 is default device</param>
        public T GetNetworkOnDevice( int deviceIdIdx ) => (deviceIdIdx == -1) ? _NetworkOnDefaultDevice : _Networks[ deviceIdIdx ];
    }
}
