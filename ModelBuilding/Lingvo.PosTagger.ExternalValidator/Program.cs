using System;

using Newtonsoft.Json;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.ExternalValidator
{
    /// <summary>
    /// 
    /// </summary>
    internal static class Program
    {
        private static void Main( string[] args )
        {
            try
            {
                var opts = OptionsExtensions.ReadInputOptions( args, "valid.json" ).opts;

                var result = Validator.Run_Validate( opts );

                var json = JsonConvert.SerializeObject( result, Formatting.None );
                PipeIPC.Client__out.Send( PipeIPC.PIPE_NAME_1, json, connectMillisecondsTimeout: 5_000 );
            }
            catch ( Exception ex )
            {
                Logger.WriteErrorLine( Environment.NewLine + ex + Environment.NewLine );

                //Console.ReadLine();
            }
            //Console.ReadLine();
        }
    }
}
