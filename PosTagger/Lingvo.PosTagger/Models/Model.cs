using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

using Lingvo.PosTagger.Utils;

namespace Lingvo.PosTagger.Models
{
    /// <summary>
    /// 
    /// </summary>
    public abstract class Model
    {
        public int EncoderEmbeddingDim { get; set; }
        public int EncoderLayerDepth { get; set; }
        public EncoderTypeEnums EncoderType { get; set; }
        public int HiddenDim { get; set; }
        public int MultiHeadNum { get; set; }
        public Vocab SrcVocab { get; set; }
        public Vocab TgtVocab { get; set; }
        //public List<Vocab> ClsVocabs { get; set; }
        //public Vocab ClsVocab
        //{
        //    get
        //    {
        //        if ( ClsVocabs == null )
        //        {
        //            ClsVocabs = new List< Vocab > { new Vocab() };
        //        }
        //        return ClsVocabs[ 0 ];
        //    }
        //    set
        //    {
        //        if ( ClsVocabs == null )
        //        {
        //            ClsVocabs = new List< Vocab > { new Vocab() };
        //        }
        //        ClsVocabs[ 0 ] = value;
        //    }
        //}
        //public bool ApplyContextEmbeddingsToEntireSequence { get; set; }

        protected Dictionary< string, float[] > __Name2Weights;
        public IReadOnlyDictionary< string, float[] > Name2Weights { get; }
        public Dictionary< string, float[] > GetName2Weights() => __Name2Weights;
        public double BestPrimaryScore { get; set; }

        public Model() { }
        public Model( int hiddenDim, int encoderLayerDepth, EncoderTypeEnums encoderType, int encoderEmbeddingDim, int multiHeadNum, Vocab srcVocab ) //, bool applyContextEmbeddingsToEntireSequence )
        {
            HiddenDim           = hiddenDim;
            EncoderLayerDepth   = encoderLayerDepth;
            EncoderType         = encoderType;
            MultiHeadNum        = multiHeadNum;
            SrcVocab            = srcVocab;
            EncoderEmbeddingDim = encoderEmbeddingDim;
            //ApplyContextEmbeddingsToEntireSequence = applyContextEmbeddingsToEntireSequence;

            __Name2Weights = new Dictionary< string, float[] >();
        }

        public void AddWeights( string name, float[] weights ) => __Name2Weights.Add( name, weights );
        public float[] GetWeights( string name )
        {
            if ( !__Name2Weights.TryGetValue( name, out var weights ) )
            {
//#if DEBUG
                //#region comm. correct.
                //*
                var n = name.Replace( "._", "." );
                if ( __Name2Weights.TryGetValue( n, out weights ) )
                {
                    return (weights);
                }

                var x = __Name2Weights.Keys.Where( k => string.Compare( k, n, true ) == 0 ).ToArray();
                if ( x.Any() && __Name2Weights.TryGetValue( x.First(), out weights ) )
                {
                    return (weights);
                }

                x = __Name2Weights.Keys.Where( k => string.Compare( k.Replace( ".m_", "." ), n, true ) == 0 ).ToArray();
                if ( x.Any() && __Name2Weights.TryGetValue( x.First(), out weights ) )
                {
                    return (weights);
                }

                var i = name.LastIndexOf( '.' );
                if ( i != -1 )
                {
                    n = name.Substring( 0, i ) + "._" + name.Substring( 0, i + 1 );
                    if ( __Name2Weights.TryGetValue( n, out weights ) )
                    {
                        return (weights);
                    }
                }
                //*/
                //#endregion
//#endif
                if ( Debugger.IsAttached )
                {
                    Debugger.Break();
                }

                Logger.WriteWarnLine( $"Weight '{name}' doesn't exist in the model." );
                return (null);
            }

            return (weights);
        }
        public void ClearWeights() => __Name2Weights.Clear();

        public void ShowModelInfo()
        {
            Logger.WriteLine( $" Encoder embedding dim: '{EncoderEmbeddingDim}'" );
            Logger.WriteLine( $"   Encoder layer depth: '{EncoderLayerDepth}'" );
            Logger.WriteLine( $"          Encoder type: '{EncoderType}'" );
            Logger.WriteLine( $"      Hidden layer dim: '{HiddenDim}'" );
            Logger.WriteLine( $"       Multi-head size: '{MultiHeadNum}'" );

            if ( SrcVocab != null ) Logger.WriteLine( $"Source vocabulary size: '{SrcVocab.Count:#,#}'" );
            if ( TgtVocab != null ) Logger.WriteLine( $"Target vocabulary size: '{TgtVocab.Count:#,#}'" );
            //if ( ClsVocabs != null )
            //{
            //    if ( ClsVocabs.Count == 1 )
            //    {
            //        Logger.WriteLine( $"Target/CLS vocabulary size: '{ClsVocabs[ 0 ].Count}'" );
            //    }
            //    else
            //    {
            //        Logger.WriteLine( $"Target/CLS vocabularies count: '{ClsVocabs.Count}' " );
            //        for ( int i = 0; i < ClsVocabs.Count; i++ )
            //        {
            //            Logger.WriteLine( $"Target/CLS vocabulary {i} size: '{ClsVocabs[ i ].Count}'" );
            //        }
            //    }
            //}
        }
    }
}
