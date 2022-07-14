using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class FeedForwardLayer : IFeedForwardLayer
    {
        private readonly WeightTensor _Whd;
        private readonly WeightTensor _Bd;
        private readonly string _Name;
        private readonly float _DropoutRatio;
        private readonly int _InputDim;
        private readonly int _OutputDim;
        private readonly int _DeviceId;
        private readonly bool _IsTrainable;

        public FeedForwardLayer( string name, int inputDim, int outputDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f )
        {
            //---Logger.WriteLine( $"Create feed forward layer '{name}' InputDim = '{inputDim}', OutputDim = '{outputDim}', DropoutRatio = '{dropoutRatio}', DeviceId = '{deviceId}'" );

            _Name         = name;
            _InputDim     = inputDim;
            _OutputDim    = outputDim;
            _DropoutRatio = dropoutRatio;
            _DeviceId     = deviceId;
            _IsTrainable  = isTrainable;

            _Whd = new WeightTensor( new long[ 2 ] { inputDim, outputDim },    deviceId, name: $"{name}.m_Whd", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor );
            _Bd  = new WeightTensor( new long[ 2 ] {        1, outputDim }, 0, deviceId, name: $"{name}.m_Bd" ,                             isTrainable: isTrainable, learningRateFactor: learningRateFactor );
        }

        public int GetDeviceId() => _DeviceId;
        public WeightTensor Process( WeightTensor input, int batchSize, ComputeGraphTensor g, float alpha = 1.0f )
        {
            WeightTensor res = g.Affine( input, _Whd, _Bd, alpha );
            var output = g.Dropout( res, batchSize, _DropoutRatio, inPlace: true );
            return (output);
        }
        public INeuralUnit CloneToDeviceAt( int deviceId ) => new FeedForwardLayer( _Name, _InputDim, _OutputDim, _DropoutRatio, deviceId, _IsTrainable );

        public List< WeightTensor > GetParams() => new List< WeightTensor > { _Whd, _Bd };
        public void Save( Model model )
        {
            _Whd.Save( model );
            _Bd.Save( model );
        }
        public void Load( Model model )
        {
            _Whd.Load( model );
            _Bd.Load( model );
        }
    }
}
