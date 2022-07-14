using System.Collections.Generic;

using Lingvo.PosTagger.Models;

namespace Lingvo.PosTagger.Network
{
    /// <summary>
    /// 
    /// </summary>
    public interface INeuralUnit
    {
        List< WeightTensor > GetParams();
        void Save( Model model );
        void Load( Model model );

        INeuralUnit CloneToDeviceAt( int deviceId );
        int GetDeviceId();
    }
}
