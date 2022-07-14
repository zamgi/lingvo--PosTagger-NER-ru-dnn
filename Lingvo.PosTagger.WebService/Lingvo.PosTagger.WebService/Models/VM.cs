using System;
using System.Collections.Generic;
using System.Linq;

namespace Lingvo.PosTagger.WebService
{
    /// <summary>
    /// 
    /// </summary>
    public readonly struct ParamsVM
    {
        public ParamsVM( string text ) : this() => Text = text;
        public string Text                 { get; init; }
        public string ModelType            { get; init; }
        public string RegimenModelType     { get; init; }
        public int?   MaxPredictSentLength { get; init; }
    }

    /// <summary>
    /// 
    /// </summary>
    public readonly struct ResultVM
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly struct TupleVM
        {
            public string Word  { get; init; }
            public string Label { get; init; }
            public override string ToString() => $"{Word} | {Label}";
        }
        /// <summary>
        /// 
        /// </summary>
        public readonly struct SentVM
        {
            public IList< TupleVM > Tuples { get; init; }
            public /*Exception*/ErrorVM Error { get; init; }
            public override string ToString() => (Error.ErrorMessage != null) ? Error.ErrorMessage : string.Join( " ", Tuples.Select( t => $"{t.Word}|{t.Label}" ) );
        }

        public IReadOnlyCollection< SentVM > Sents { get; init; }
        public ErrorVM Error { get; init; }
        public override string ToString() => string.Join( "\r\n", Sents );
    }

    /// <summary>
    /// 
    /// </summary>
    public readonly struct ErrorVM
    {
        public ErrorVM( Exception ex )
        {
            ErrorMessage     = ex?.Message;
            FullErrorMessage = ex?.ToString();
        }
        public string ErrorMessage     { get; init; }
        public string FullErrorMessage { get; init; }
        public override string ToString() => (ErrorMessage ?? string.Empty);
    }
}
