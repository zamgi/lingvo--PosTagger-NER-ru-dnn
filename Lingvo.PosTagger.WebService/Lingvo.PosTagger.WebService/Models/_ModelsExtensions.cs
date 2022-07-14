using System;
using System.Collections.Generic;
using System.Linq;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.WebService
{
    /// <summary>
    /// 
    /// </summary>
    internal static class ModelsExtensions
    {
        [M(O.AggressiveInlining)] public static ResultVM ToResultVM( this in (List< (IList< ResultVM.TupleVM > result, Exception error) > results, Exception error) t ) => 
            new ResultVM() { Error = new ErrorVM( t.error ), Sents = t.results?.Select( x => x.ToPosTaggerResultVM() ).ToList() };

        [M(O.AggressiveInlining)] public static ResultVM.SentVM ToPosTaggerResultVM( this in (IList< ResultVM.TupleVM > results, Exception error) t ) => 
            new ResultVM.SentVM() { Error = new ErrorVM( t.error ), Tuples = t.results };

        [M(O.AggressiveInlining)] public static ErrorVM ToErrorVM( this Exception ex ) => new ErrorVM() { ErrorMessage = ex.Message, FullErrorMessage = ex.ToString(), };

#if DEBUG
        public static string ToText( this in ParamsVM p )
        {
            if ( (p.Text != null) && (250 < p.Text.Length) )
            {
                return (p.Text.Substring( 0, 250 ) + "...");
            }
            return (p.Text);
        }
#endif
    }
}
